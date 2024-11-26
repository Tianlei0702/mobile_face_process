import sqlite3
import numpy as np
import ujson

class FaceDB():
    def __init__(self, db_path='facedb.db'):
        self.conn = sqlite3.connect(db_path)
        self.c = self.conn.cursor()
        self.c.execute('''CREATE TABLE IF NOT EXISTS face
                    (id INTEGER PRIMARY KEY,
                    name TEXT, 
                    features LIST)''')

    def add_face(self, name, feature):
        features_json = ujson.dumps(list(feature))
        self.c.execute("INSERT INTO face (name, features) VALUES (?,?)", (name, features_json))
        self.conn.commit()
    
    def check_face(self):
        self.c.execute("SELECT * FROM face")
        rows = self.c.fetchall()
        print(rows)

    def get_face_feature(self):
        self.c.execute("SELECT * FROM face")
        rows = self.c.fetchall()
        names = []
        img_features = []
        for row in rows:
            name = row[1]
            names.append(name)
            features_str = row[2]
            features = np.array(ujson.loads(features_str))
            img_features.append(features)
        return names, img_features

    def remove_face(self,name):
        self.c.execute("DELETE FROM face WHERE name =?", (name,))
        self.conn.commit()

    def remove_all(self):
        self.c.execute('DELETE FROM face')
        self.conn.commit()






if __name__=="__main__":
    db = FaceDB('facedb.db')
    #db.add_face(name = 'Job', feature=np.array([1,1,1]))
    db.get_face_feature()

