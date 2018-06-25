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

    def close_spider(self, spider):
        self.conn.commit()
        self.conn.close()

    def process_item(self, item, spider):
        word = item['term']
        index = self.cursor.execute('SELECT entry FROM entry_etymologies')
        new_index = max([item[0] for item in self.cursor.fetchall()] + [-1]) + 1
        self.cursor.execute(f'INSERT INTO entry_etymologies(entry, etymology) VALUES ({new_index}, "{word}")')
        return item
