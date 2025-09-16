class Log:
    def __init__(self, db):
        self.db = db
        
    def insert(self, data:dict):
        self.insert_id = self.db.insert(data)
        
    def update(self, data:dict):
        self.db.update(data, doc_ids=[self.insert_id])
        