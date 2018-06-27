import scrapy, re, pandas as pd
from bs4 import BeautifulSoup

class WiktionarySpider(scrapy.Spider):
    name = "wiktionary_spider"

    def start_requests(self):
        all_pos = ['adfix', 'adjective', 'adnoun', 'adverb', 'article', 'auxiliary verb', 'cardinal number', 'collective numeral', 
           'conjunction', 'coverb', 'demonstrative determiner', 'demonstrative pronoun', 'determinative', 'determiner', 
           'gerund', 'indefinite pronoun', 'infinitive', 'interjection', 'interrogative pronoun', 'intransitive verb', 
           'noun', 'number', 'numeral', 'ordinal', 'ordinal number', 'part of speech', 'participle', 'particle', 
           'personal pronoun', 'phrasal preposition', 'possessive adjective', 'possessive determiner', 'possessive pronoun', 
           'postposition', 'preposition', 'preverb', 'pronoun', 'quasi-adjective', 'reciprocal pronoun', 'reflexive pronoun', 
           'relative pronoun', 'speech disfluency', 'substantive', 'transitive', 'transitive verb', 'verb', 'verbal noun']
        
        # Option #2 for word-language pairs
        #cursor.execute('SELECT word, language_name \
        #                FROM etymologies e, languages l\
        #                WHERE e.language_code = l.language_code AND _id NOT IN (SELECT etymology_id FROM entry_connections) \
        #                limit 10')
        #wls = [[w.decode('utf-8','replace').strip(), l.decode()] for w, l in cursor.fetchall()]; wls
        #url_terms = [row[0] if not row[1].startswith('Proto') else 'Reconstruction:'+row[1]+'/'+row[0] for row in wls]; url_terms
        #for url in set(url_terms):
        #print(url)
        
     # Get the list of word-language pairs
        df = pd.read_csv('~/etymology_files/ety_master.csv', 
                 usecols = ['word', 'language'], 
                 converters={'word' : str, 'language': str})

        # Set all non-reconstructions to be 'None'
        normal_language_rows = [not language.startswith('Proto') for language in df['language'].tolist()] #bools
        df.loc[normal_language_rows, 'language'] = None
        df = df.drop_duplicates()
        url_terms = [row[0] if row[1] is None else 'Reconstruction:'+row[1]+'/'+row[0] for row in df.values]
        urls = ['https://en.wiktionary.org/api/rest_v1/page/html/' + term for term in url_terms]
        
        for url in urls:
            yield scrapy.Request(url=url, meta={'all_pos': all_pos}, callback=self.parse)
        
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
        
        all_pos = response.meta['all_pos']
        
        soup = BeautifulSoup(response.body, 'html.parser')
        
        page = response.url.replace('https://en.wiktionary.org/api/rest_v1/page/html/', '')
        page = re.sub('Reconstruction:[^\/]+?\/(.*)', r'\1', page) #Remove reconstruction text if necessary
        page_data = {'term': page} # Variable to store the data

        for lang_node in soup.find_all('h2'): # Go through each Language
            language = lang_node.text.strip()
            language_entries = [{}]

            for ety_pronunc_pos_node in lang_node.parent.find_all('h3'): # Go through each ety,pronu, or pos
                node_data = []
                node_class = re.sub('_\d+| \d+', '', ety_pronunc_pos_node.text).lower() #removed '_x' info

                if node_class == 'etymology':
                    #Only looking at the first <p> element for etymology text
                    p_node = ety_pronunc_pos_node.parent.find('p')
                    if p_node is not None: 
                        entry_data = {'etymology': p_node.text}
                    else:
                        continue

                    for sub_ety_pos in ety_pronunc_pos_node.parent.find_all('h4'):
                        if sub_ety_pos.text.lower() in all_pos:
                            defs = getDefsFromPOS(sub_ety_pos)
                            if defs is None:
                                continue
                            else:
                                entry_data[sub_ety_pos.text.lower()] = defs

                    if any(['etymology' in entry for entry in language_entries]): #If an etymology already exists add to new entry
                        language_entries.append(entry_data)
                    else:
                        language_entries[0].update(entry_data)

                # Need to see if there are sub items of this etymology
                elif node_class == 'pronunciation':
                    ipa_nodes = ety_pronunc_pos_node.parent.select('span.IPA') #dataquest.io/blog/web-scraping-tutorial-python/
                    if ipa_nodes: #Only add pronunciation if there are ipa_nodes
                        node_data = ipa_nodes[0].text
                        language_entries[0]['pronunciation'] = node_data

                elif node_class in all_pos: # Here are the definitions
                    defs = getDefsFromPOS(ety_pronunc_pos_node)
                    if defs is None:
                        continue
                    else:
                        language_entries[0][node_class] = defs

                else: # Skip all other node_classes
                    continue

                page_data[language] = language_entries #Add all the language entries to the language 

        yield page_data
