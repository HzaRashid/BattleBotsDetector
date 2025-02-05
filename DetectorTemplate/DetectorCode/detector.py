from abc_classes import ADetector
from teams_classes import DetectionMark
import random
import pandas as pd
import json

class Detector(ADetector):
    def detect_bot(self, session_data):
        self.process_data(session_data)

        # todo logic
        # Example:
        marked_account = []
        
        for user in session_data.users:
            prediction = random.choice((True, False))
            # print(prediction)
            
            marked_account.append(DetectionMark(user_id=user['id'], confidence=50, bot=prediction))

        return marked_account
    
    def process_data(self, session_data):
        print(session_data.session_id)
        print(session_data.lang)
        print(session_data.metadata)
        print(session_data.posts)
        print(session_data.users)