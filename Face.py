from deepface import DeepFace
import pandas as pd
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
df2 = pd.DataFrame()
for i in range(9):
    df = DeepFace.find(img_path = '/Users/akashvarun/Desktop/Face Verification/photos/d2.png', db_path ='/Users/akashvarun/Desktop/Face Verification/photos/', model_name = models[i])
    df2 = df2.append(df)
df2 = df2.drop_duplicates(subset = "identity")
df2.to_csv('file2.csv')
