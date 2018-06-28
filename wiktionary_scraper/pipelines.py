# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import mysql.connector, os

class WiktionaryScraperPipeline(object):

    collection_name = 'scrapy_items'

    def __init__(self):
        self.user = os.environ['MYSQL_DB_USER']
        self.password = os.environ['MYSQL_DB_PASSWORD']
        self.host = '127.0.0.1'
        self.database = 'etymology_explorer_staging'

    @classmethod
    def from_crawler(cls, crawler):
        return cls()

    def open_spider(self, spider):
        self.conn = mysql.connector.connect(user=self.user, password=self.password, host=self.host, database=self.database)
        self.cursor = self.conn.cursor()
        self.cursor.execute('SET NAMES utf8mb4;') #To avoid unicode issues

    def close_spider(self, spider):
        self.conn.commit()
        self.conn.close()

    def process_item(self, item, spider):
        
        all_pos = ['adfix', 'adjective', 'adnoun', 'adverb', 'article', 'auxiliary verb', 'cardinal number', 'collective numeral',
           'conjunction', 'coverb', 'demonstrative determiner', 'demonstrative pronoun', 'determinative', 'determiner',
           'gerund', 'indefinite pronoun', 'infinitive', 'interjection', 'interrogative pronoun', 'intransitive verb',
           'noun', 'number', 'numeral', 'ordinal', 'ordinal number', 'part of speech', 'participle', 'particle',
           'personal pronoun', 'phrasal preposition', 'possessive adjective', 'possessive determiner', 'possessive pronoun',
           'postposition', 'preposition', 'preverb', 'pronoun', 'quasi-adjective', 'reciprocal pronoun', 'reflexive pronoun',
           'relative pronoun', 'speech disfluency', 'substantive', 'transitive', 'transitive verb', 'verb', 'verbal noun']
        
        def getNewKey(column, table):
            self.cursor.execute(f'SELECT max({column}) FROM {table}')
            max_entry_id = self.cursor.fetchone()[0]; 
            new_entry_id = max_entry_id + 1 if max_entry_id is not None else 0
            return new_entry_id
        
        def insert(table, **kwargs):
            columns = [str(k) for k,v in kwargs.items()]
            col_text = '('+', '.join(columns)+')'
            
            values = [str(v) if type(v) != str else repr(v) for k,v in kwargs.items()]
            val_text = '('+', '.join(values)+')'
            
            self.cursor.execute(f'INSERT INTO {table}{col_text} VALUES {val_text}')
        
        word = item['term'] # Get the word for these entries

        for key, entries in item.items():
            if key == 'term': continue
            language = key

            self.cursor.execute(
                f'SELECT _id FROM etymologies e, languages l \
                    WHERE word = "{word}" and language_name = "{language}" and e.language_code = l.language_code')
            ety_id_result = self.cursor.fetchone()
            
            if ety_id_result is not None: 
                ety_id = ety_id_result[0]
            else:
                ety_id = getNewKey('_id', 'etymologies') + 1
                self.cursor.execute(
                    f'INSERT INTO etymologies(_id, word, language_code) \
                        SELECT {ety_id}, "{word}", language_code FROM languages WHERE language_name = "{language}"')
                
            for entry in entries:
                # Get a new entry key, make the entry connection
                new_entry_id = getNewKey('entry_id', 'entry_connections')
                insert('entry_connections', etymology_id=ety_id, entry_id=new_entry_id)

                for node_key, node_value in entry.items():
                    if node_key == 'pronunciation':
                        insert('entry_pronunciations', pronunciation=node_value, entry_id=new_entry_id)

                    elif node_key == 'etymology':
                        insert('entry_etymologies', etymology=node_value, entry_id=new_entry_id)

                    elif node_key in all_pos:
                        new_pos_key = getNewKey('pos_id', 'entry_pos')
                        insert('entry_pos', pos_name=node_key, pos_id=new_pos_key, entry_id=new_entry_id)

                        for definition in node_value:
                            insert('entry_definitions', definition=definition, pos_id=new_pos_key)
        self.conn.commit()
        return item
