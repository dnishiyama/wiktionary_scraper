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
		self.staging_database = 'etymology_explorer_staging'
		self.live_database = 'etymology_explorer'
		self.all_pos = None

	@classmethod
	def from_crawler(cls, crawler):
		return cls()

	def open_spider(self, spider, live=False):
		database = self.live_database if live else self.staging_database
		self.conn = mysql.connector.connect(user=self.user, password=self.password, host=self.host, database=database)
		self.cursor = self.conn.cursor()
		self.cursor.execute('SET NAMES utf8mb4;') #To avoid unicode issues

		self.all_pos = spider.all_pos

	def close_spider(self, spider):
		self.conn.commit()
		self.conn.close()

	def process_item(self, item, spider):	
		word = item['term'] # Get the word for these entries
		
		for key, entries in item.items():
			if key in ('term', 'status'): continue
			language = key

			sql = f'SELECT _id FROM etymologies WHERE word = {repr(word)} and language_name = {repr(language)}'
			ety_id_result = self.execute_sql(sql)
			
			if ety_id_result: 
				ety_id = ety_id_result[0][0]
			else:
				ety_id = self.getNewKey('_id', 'etymologies')
				self.cursor.execute(f'INSERT INTO etymologies(_id, word, language_name) VALUES ({ety_id}, {repr(word)}, {repr(language)})')
			
			# Get all the existing entries to compare for updates / leaving alone
			self.cursor.execute(f'SELECT entry_id, entry_number FROM entry_connections WHERE etymology_id = {ety_id}')
			existing_entries = self.cursor.fetchall()

			for entry_number, entry in enumerate(entries):

				# Get the matching entry_id based on entry_number
				matching_entry = [row[0] for row in existing_entries if row[1] == entry_number + 1]

				if matching_entry: #If the entries already exist, delete the existing (except the etymology) update etymologies if they are different
					new_entry_id = matching_entry[0]

					#Deleting existing entry information (except for etymology data)
					self.cursor.execute(f'DELETE ed, ep FROM entry_definitions ed JOIN entry_pos ep ON ed.pos_id = ep.pos_id WHERE ep.entry_id = {new_entry_id}')
					self.cursor.execute(f'DELETE FROM entry_pronunciations WHERE entry_id = {new_entry_id}')

				else: # If the entry number doesn't exist, then make a new entry for it 
				# Get a new entry key, make the entry connection
					new_entry_id = self.getNewKey('entry_id', 'entry_connections')
					self.insert('entry_connections', etymology_id=ety_id, entry_number=entry_number+1, entry_id=new_entry_id)
				
				# Now self.insert all the new entry data
				for node_key, node_value in entry.items():
					if node_key == 'pronunciation':
						self.insert('entry_pronunciations', pronunciation=node_value, entry_id=new_entry_id)

					#Special case, update if it is different, with the new_connections flag as 1
					elif node_key == 'etymology':

						# If an etymology already exists for this entry, and it is the same, update, otherwise skip
						existing_etymology = self.execute_sql(f'SELECT etymology, lock_code FROM entry_etymologies WHERE entry_id = {new_entry_id}')
						if existing_etymology and existing_etymology[0][0].decode() != node_value and existing_etymology[0][1] == 0: #and it isn't locked
							self.insert('entry_etymologies', replace=True, 
								etymology=node_value, entry_id=new_entry_id, new_connections=1, connection_code='None', lock_code = 0)
							self.generateConnections(new_entry_id, node_value)
						elif not existing_etymology: 
							self.insert('entry_etymologies', 
								etymology=node_value, entry_id=new_entry_id, new_connections=1, connection_code='None', lock_code = 0)
							self.generateConnections(new_entry_id, node_value)

					elif node_key in self.all_pos:
						new_pos_key = self.getNewKey('pos_id', 'entry_pos')
						self.insert('entry_pos', pos_name=node_key, pos_id=new_pos_key, entry_id=new_entry_id)

						for definition in node_value:
							self.insert('entry_definitions', definition=definition, pos_id=new_pos_key)
		return item

	def generateConnections(self, entry_id, etymology):
		#Make the connection code from the etymology
		#Insert that connection code into the MYSQL @ that entry_id, lock_code still 0, new_connections = 0
		#Check for errors in the connection code
		#If there are none, generate the connections from the connection code
		#Add the connections and the connection sources to the MYSQL database
		print('Generating connections for', entry_id, etymology);

	def execute_sql(self, sql, debug=False):
		"""Returns the fetchall() result after error catching"""
		try:
			self.cursor.execute(sql)
		except Exception as e:
			self.cursor.fetchall(); #Fetch to prevent this error from cascading
			raise (e) # Raise so that no errors are introduced to the MYSQL Database
		return self.cursor.fetchall()
	
	def getNewKey(self, column, table):
		new_entry_id = self.execute_sql(f"""SELECT MIN(t1.{column})
			FROM(
				SELECT 1 AS {column}
				UNION ALL
				SELECT {column} + 1
				FROM {table}
			) t1
			LEFT OUTER JOIN {table} t2
			ON t1.{column} = t2.{column}
			WHERE t2.{column} IS NULL;""")[0][0]
		#max_entry_id = self.execute_sql(f'SELECT max({column}) FROM {table}')[0][0]
		#new_entry_id = max_entry_id + 1 if max_entry_id is not None else 0
		return new_entry_id
	
	def insert(self, table, replace=False, **kwargs):
		insert = 'REPLACE' if replace else 'INSERT' 
		columns = [str(k) for k,v in kwargs.items()]
		col_text = '('+', '.join(columns)+')'
		
		values = [str(v) if type(v) != str else repr(v) for k,v in kwargs.items()]
		val_text = '('+', '.join(values)+')'
		
		self.cursor.execute(f'{insert} INTO {table}{col_text} VALUES {val_text}')
