import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
from datetime import datetime
import shutil

class VectorDB:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        self.metadata = []
        
        # กำหนด path แบบ absolute
        self.root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.data_dir = os.path.join(self.root_dir, 'data')
        self.db_file = os.path.join(self.data_dir, 'vector_db.pkl')
        
        print(f"Initializing vector database...")
        print(f"Root directory: {self.root_dir}")
        print(f"Data directory: {self.data_dir}")
        print(f"Database file: {self.db_file}")
        
        # สร้างโฟลเดอร์ data ถ้ายังไม่มี
        try:
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
                print(f"Created data directory at {self.data_dir}")
            
            # ทดสอบการเขียนไฟล์
            test_file = os.path.join(self.data_dir, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print("Write permission test: Passed")
            
            if os.path.exists(self.db_file):
                self.load_db()
            else:
                # สร้างไฟล์ DB ใหม่
                self.save_db()
                print(f"Created new vector database at {self.db_file}")
        
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise Exception(f"Failed to initialize vector database: {str(e)}")

        print(f"Vector database path: {self.db_file}")
        print(f"Current database size: {len(self.texts)} texts")

    def add_text(self, text: str, metadata=None):
        vector = self.encoder.encode([text])[0]
        self.index.add(np.array([vector]).astype('float32'))
        
        # Add metadata
        meta = {
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'additional_info': metadata
        }
        self.metadata.append(meta)
        self.texts.append(text)
        self.save_db()  # เรียก save หลังจากเพิ่มข้อมูล
        print(f"Added text to vector database: {text}")
        if len(self.texts) % 10 == 0:  # แสดงสถานะทุก 10 รายการ
            print(f"Database size: {len(self.texts)} texts")

    def search(self, query: str, k: int = 5):
        if len(self.texts) == 0:
            print("Vector database is empty")
            return []
            
        query_vector = self.encoder.encode([query])[0]
        distances, indices = self.index.search(np.array([query_vector]).astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                similarity = float(1 / (1 + distances[0][i]))
                result = {
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'similarity_score': similarity
                }
                results.append(result)
                print(f"Found similar text (score: {similarity:.2f}): {self.texts[idx][:100]}...")
                
        return results

    def save_db(self):
        try:
            # สร้าง backup ก่อนถ้ามีไฟล์เดิม
            if os.path.exists(self.db_file):
                backup_file = self.db_file + '.bak'
                shutil.copy2(self.db_file, backup_file)
            
            # บันทึกไฟล์ใหม่
            with open(self.db_file, 'wb') as f:
                data = {
                    'texts': self.texts,
                    'metadata': self.metadata,
                    'index': faiss.serialize_index(self.index)
                }
                pickle.dump(data, f)
            
            print(f"Successfully saved database with {len(self.texts)} texts")
            
            # ลบ backup ถ้าบันทึกสำเร็จ
            if os.path.exists(self.db_file + '.bak'):
                os.remove(self.db_file + '.bak')
                
            return True
            
        except Exception as e:
            print(f"Error saving database: {str(e)}")
            # กู้คืนจาก backup ถ้ามี
            if os.path.exists(self.db_file + '.bak'):
                shutil.copy2(self.db_file + '.bak', self.db_file)
                os.remove(self.db_file + '.bak')
            return False

    def load_db(self):
        try:
            if not os.path.exists(self.db_file):
                print(f"No existing database found at {self.db_file}")
                return False
                
            with open(self.db_file, 'rb') as f:
                data = pickle.load(f)
                self.texts = data['texts']
                self.metadata = data.get('metadata', [])
                self.index = faiss.deserialize_index(data['index'])
                print(f"Successfully loaded database with {len(self.texts)} texts")
                return True
        except Exception as e:
            print(f"Error loading database: {str(e)}")
            return False
