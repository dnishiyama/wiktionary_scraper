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
        

	# Get the list of word-language pairs
        df = pd.read_csv('~/etymology_files/ety_master.csv', 
                 usecols = ['word', 'language'], 
                 converters={'word' : str, 'language': str})

        # Set all non-reconstructions to be 'None'
        normal_language_rows = [not language.startswith('Proto') for language in df['language'].tolist()] #bools
        df.loc[normal_language_rows, 'language'] = None
        df = df.drop_duplicates()
        url_terms = [row[0] if row[1] is None else 'Reconstruction:'+row[1]+'/'+row[0] for row in df.values]
        urls = ['https://en.wiktionary.org/api/rest_v1/page/html/' + term for term in url_terms[:5]]
        
        for url in urls:
            yield scrapy.Request(url=url, meta={'all_pos': all_pos}, callback=self.parse)

    def parse(self, response):
        all_pos = response.meta['all_pos']
        page = response.url.replace('https://en.wiktionary.org/api/rest_v1/page/html/', '')
        page = re.sub('Reconstruction:[^\/]+?\/(.*)', r'\1', page) #Remove reconstruction text if necessary

        soup = BeautifulSoup(response.body, 'html.parser')

        page_data = {'term': page} # Variable to store the data

        for lang_node in soup.find_all('h2'): # Go through each Language
            language = lang_node.text.strip()

            for ety_pronunc_pos_node in lang_node.parent.find_all('h3'): # Go through each ety,pronu, or pos
                node_data = []
                node_class = re.sub('_\d+| \d+', '', ety_pronunc_pos_node.text).lower()
                node_text = ety_pronunc_pos_node.text.lower()
                #print(node_class, node_text)

                if node_class == 'etymology':
                    node_data = ety_pronunc_pos_node.parent.find('p').text #Only looking at the first <p> element

                elif node_class == 'pronunciation':
                    ipa_nodes = ety_pronunc_pos_node.parent.select('span.IPA') #dataquest.io/blog/web-scraping-tutorial-python/
                    if ipa_nodes: node_data = ipa_nodes[0].text

                elif node_class in all_pos: # Here are the definitions
                    for li in list(ety_pronunc_pos_node.parent.find('ol').children): # get defs from ordered list
                        if li.name != 'li': continue # This is a newline tag

                        if li.find('ol'): #Ordered list means the sub items are the definition
                            for sub_li in list(li.find('ol').children):
                                if sub_li.name != 'li': continue # Skip newline tags

                                for child in sub_li.children:
                                    if child.name not in ('a', 'span', None): child.clear() #Get rid of quotes and subitems

                                node_data.append(sub_li.text.strip())

                        else: # otherwise grab the text

                            for child in li.children:
                                if child.name not in ('a', 'span', None): child.clear() #Get rid of quotes and subitems
                            node_data.append(li.text.strip())

                else: # Skip all other node_classes
                    continue

                page_data.setdefault(language, {})[node_text] = node_data

        yield page_data
