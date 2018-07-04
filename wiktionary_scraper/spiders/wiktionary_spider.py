import scrapy, re, pandas as pd, mysql.connector, os
from bs4 import BeautifulSoup
from scrapy.spidermiddlewares.httperror import HttpError
from twisted.internet.error import DNSLookupError
from twisted.internet.error import TimeoutError, TCPTimedOutError

class WiktionarySpider(scrapy.Spider):
	name = "wiktionary_spider"
	user = os.environ['MYSQL_DB_USER']
	password = os.environ['MYSQL_DB_PASSWORD']
	host = '127.0.0.1'
	database = 'etymology_explorer_staging'
	conn = None; cursor = None
	all_pos = ['adfix', 'adjective', 'adnoun', 'adverb', 'article', 'auxiliary verb', 'cardinal number', 'collective numeral', 
		'conjunction', 'coverb', 'demonstrative determiner', 'demonstrative pronoun', 'determinative', 'determiner', 
		'gerund', 'indefinite pronoun', 'infinitive', 'interjection', 'interrogative pronoun', 'intransitive verb', 
		'noun', 'number', 'numeral', 'ordinal', 'ordinal number', 'part of speech', 'participle', 'particle', 
		'personal pronoun', 'phrasal preposition', 'possessive adjective', 'possessive determiner', 'possessive pronoun', 
		'postposition', 'preposition', 'preverb', 'pronoun', 'quasi-adjective', 'reciprocal pronoun', 'reflexive pronoun', 
		'relative pronoun', 'speech disfluency', 'substantive', 'transitive', 'transitive verb', 'verb', 'verbal noun', 
		'infix', 'suffix', 'prefix', 'root'] # Last 4 needed for reconstructions

	def __init__(self, *a, **kw):
		super(WiktionarySpider, self).__init__(*a, **kw)
		self.conn = mysql.connector.connect(user=self.user, password=self.password, host=self.host, database=self.database)
	
	# Function called by scrapy.	
	def start_requests(self):	
		self.cursor = self.conn.cursor()
		self.cursor.execute('SET NAMES utf8mb4;') #To avoid unicode issues
		
		self.cursor.execute('SELECT * FROM languages')
		lc2ln = {row[1].decode(): row[0].decode() for row in self.cursor.fetchall()}; lc2ln['alu']
		
		self.cursor.execute('SELECT word, language_code FROM etymologies WHERE _id NOT IN (SELECT DISTINCT etymology_id FROM entry_connections)')
		new_terms = [[row[0].decode().strip(), row[1].decode()] for row in self.cursor.fetchall()]
		
		url_terms = [row[0] if not row[1].endswith('-pro') else 'Reconstruction%3A'+lc2ln[row[1]]+'%2F'+row[0] for row in new_terms]
		
		#Get bad urls (suffices) that have received 404 responses
		self.cursor.execute('SELECT url_suffix FROM wiktionary_page_dne')
		bad_url_suffices = set([row[0] for row in self.cursor.fetchall()])

		for url_suffix in set(url_terms):
			if url_suffix in bad_url_suffices: 
				#self.logger.info('Skipping ' + url_suffix + ' due to past 404')
				continue #skip urls that have been 404s in the past
			url = 'https://en.wiktionary.org/api/rest_v1/page/html/' + url_suffix
			term = re.sub('Reconstruction%3A.+?%2F(.*)', r'\1', url_suffix) #Remove reconstruction text if necessary
			yield scrapy.Request(url=url, meta={'term':term, 'url_suffix': url_suffix}, callback=self.parse, errback=self.errback)
		
	def parse(self, response):
		
		# Must have this function within this context
		def getDefsFromPOS(ety_pronunc_pos_node):
			keep_def_tags = ('i', 'b', 'a', 'span', None)
			node_data = []
			if ety_pronunc_pos_node.parent.find('ol') is None: return None
			for li in list(ety_pronunc_pos_node.parent.find('ol').children): # get defs from ordered list
				if li.name != 'li': continue # This is a newline tag

				if li.find('ol'): #Ordered list means the sub items are the definition
					for sub_li in list(li.find('ol').children):
						if sub_li.name != 'li': continue # Skip newline tags

						for child in sub_li.children:
							if child.name not in keep_def_tags: child.clear() #Get rid of quotes and subitems

						node_data.append(sub_li.text.strip())

				else: # otherwise grab the text
					for child in li.children:
						if child.name not in keep_def_tags: child.clear() #Get rid of quotes and subitems
					node_data.append(li.text.strip())
			return node_data
		
		page_data = {'term': response.meta['term'], 'status': response.status} # Variable to store the data
		# Check for 404 and save word in database. Save to wiktionary_page_dne if MYSQL is 404

		soup = BeautifulSoup(response.body, 'html.parser')
		
		for lang_node in soup.find_all('h2'): # Go through each Language
			language = lang_node.text.strip()
			language_entries = [{}] 
			entry_number = 1 #List starts at 1 to match wiktionary

			for ety_pronunc_pos_node in lang_node.parent.find_all('h3'): # Go through each ety,pronu, or pos
				node_class = re.sub('_\d+| \d+', '', ety_pronunc_pos_node.text).lower() #removed '_x' info

				if node_class == 'etymology': #Have to assume that etymologies are the first item of new entries if there are multipl (ety can be any of only entry)
					if entry_number != 1 or 'etymology' in language_entries[0]:
						entry_number = len(language_entries) + 1
						language_entries.append({})
				
					# If the entry number is 1, then need to add to it if there is no etymology, or if there is an etymology, then increment the entry_number
					# if entry number is not 1 then increment it
					# Set entry number based on the number of etymologies saved (could be set based on etymology_1 name)

					#Only looking at the first <p> element for etymology text
					p_node = ety_pronunc_pos_node.parent.find('p')
					if p_node is not None: 
						entry_data = {'etymology': p_node.text.replace('\u200e', '')}
					else:
						continue

					for sub_ety_pos in ety_pronunc_pos_node.parent.find_all('h4'):
						if sub_ety_pos.text.lower() in self.all_pos:
							defs = getDefsFromPOS(sub_ety_pos)
							if defs is None:
								continue
							else:
								entry_data[sub_ety_pos.text.lower()] = defs

					#if any(['etymology' in entry for entry in language_entries]): #If an etymology already exists add to new entry
					#	language_entries.append(entry_data)
					#else:
					language_entries[entry_number - 1].update(entry_data) #Set the entry number with this etymology data

				# Need to see if there are sub items of this etymology
				elif node_class == 'pronunciation':
					ipa_nodes = ety_pronunc_pos_node.parent.select('span.IPA') #dataquest.io/blog/web-scraping-tutorial-python/
					if ipa_nodes: #Only add pronunciation if there are ipa_nodes
						language_entries[entry_number - 1]['pronunciation'] = ipa_nodes[0].text

				elif node_class in self.all_pos: # Here are the definitions
					defs = getDefsFromPOS(ety_pronunc_pos_node)
					if defs is None:
						continue
					else:
						language_entries[entry_number - 1][node_class] = defs

				else: # Skip all other node_classes
					continue

				page_data[language] = language_entries #Add all the language entries to the language 

		yield page_data

	def errback(self, failure):
		# log all failures
		# self.logger.error(repr(failure))

		# in case you want to do something special for some errors,
		# you may need the failure's type:

		if failure.check(HttpError):
			# these exceptions come from HttpError spider middleware
			# you can get the non-200 response
			response = failure.value.response
			term = response.meta['term']
			if response.status != 404:
				self.logger.error('HttpError on %s', response.url)
			else:
				self.cursor.execute(f'INSERT IGNORE INTO wiktionary_page_dne VALUES({repr(term)})')

		elif failure.check(DNSLookupError):
			# this is the original request
			request = failure.request
			self.logger.error(repr(failure))
			self.logger.error('DNSLookupError on %s', request.url)

		elif failure.check(TimeoutError, TCPTimedOutError):
			request = failure.request
			self.logger.error(repr(failure))
			self.logger.error('TimeoutError on %s', request.url)
		else:
			self.logger.error('Unknown Error!')
			self.logger.error(repr(failure))

	def closed(self, reason):
		self.conn.commit()
		self.conn.close()
