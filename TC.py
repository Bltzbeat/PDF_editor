import os
import pandas as pd
from pathlib import Path
import io
import numpy as np
import re
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import torch
from openpyxl import load_workbook
import openpyxl
from openpyxl.styles import Border, Side, PatternFill, Font, Alignment

class TextAnalyzer:
    def __init__(self):
        # Initialize model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = SentenceTransformer('paraphrase-MPNet-base-v2', device=self.device)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Attempting to use fallback model...")
            self.model = SentenceTransformer('distilbert-base-nli-mean-tokens', device=self.device)

    def create_embeddings(self, texts, batch_size):
        text_map = {i: str(text) for i, text in enumerate(texts) if text and not pd.isna(text) and not str(text).isspace()}
        unique_texts = list(set(text_map.values()))
        
        embeddings = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i in range(0, len(unique_texts), batch_size):
                batch = unique_texts[i:i+batch_size]
                batch_embeddings = self.model.encode(batch, show_progress_bar=False)
                embeddings.extend(batch_embeddings)
        
        text_to_embedding = {text: emb for text, emb in zip(unique_texts, embeddings)}
        embedding_dim = embeddings[0].shape[0] if embeddings else self.model.get_sentence_embedding_dimension()
        return np.array([text_to_embedding[text_map[i]] if i in text_map 
                        else np.zeros(embedding_dim) for i in range(len(texts))])

    def calculate_cosine_similarity(self, embeddings, batch_size=1000):
        n = len(embeddings)
        cosine_sim = np.zeros((n, n))
        
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            batch = embeddings[i:end]
            cosine_sim[i:end] = cosine_similarity(batch, embeddings)
            del batch
            
        return cosine_sim

    def perform_topic_clustering(self, input_file: str, text_column: str, output_dir: str, threshold1: float, threshold2: float, threshold3: float, batch_size: int = 500):
        try:
            df = pd.read_excel(input_file)
            temp_dir = os.path.join(output_dir, "temp_clustering")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
                
            print("Creating embeddings...")
            texts = df[text_column].fillna('').tolist()
            embeddings = self.create_embeddings(texts, batch_size)
            
            print("Calculating cosine similarity...")
            cosine_sim = self.calculate_cosine_similarity(embeddings, batch_size=batch_size)
            
            topic_data = {
                threshold1: {'topics': [], 'titles': [], 'counts': {}},
                threshold2: {'topics': [], 'titles': [], 'counts': {}},
                threshold3: {'topics': [], 'titles': [], 'counts': {}}
            }
            
            # Process for each threshold
            for threshold in [threshold1, threshold2, threshold3]:
                print(f"\nProcessing for similarity threshold: {threshold}")
                
                topics = [-1] * len(df)
                current_topic = 1
                
                empty_topic = 0
                for i in range(len(df)):
                    if pd.isna(df.iloc[i][text_column]) or str(df.iloc[i][text_column]).isspace():
                        topics[i] = empty_topic
                
                for i in range(len(df)):
                    if topics[i] == -1:
                        topics[i] = current_topic
                        break
                
                print("Assigning topics...")
                for i in tqdm(range(1, len(df)), desc="Assigning topics"):
                    if topics[i] != -1:
                        continue
                        
                    found_similar = False
                    for j in range(i):
                        if topics[j] > 0 and cosine_sim[i][j] >= threshold:
                            topics[i] = topics[j]
                            found_similar = True
                            break
                    
                    if not found_similar:
                        current_topic += 1
                        topics[i] = current_topic
                
                df[f'Topics_threshold_{threshold}'] = [f"Empty Content" if x == 0 else f"Topic {x}" if x != -1 else "Unassigned" for x in topics]
                
                print("Generating topic titles...")
                df[f'Topic_Title_threshold_{threshold}'] = ''
                topic_titles = {}
                topic_counts = {}
                
                topic_titles["Empty Content"] = "Empty Content"
                topic_counts["Empty Content"] = topics.count(0)
                
                for topic_num in range(1, current_topic + 1):
                    try:
                        topic_name = f"Topic {topic_num}"
                        topic_texts = df[df[f'Topics_threshold_{threshold}'] == topic_name][text_column]
                        
                        topic_counts[topic_name] = len(topic_texts)
                        
                        if not topic_texts.empty:
                            topic_embeddings = self.model.encode(topic_texts.tolist(), show_progress_bar=False)
                            avg_embedding = np.mean(topic_embeddings, axis=0)
                            similarities = cosine_similarity([avg_embedding], topic_embeddings)[0]
                            top_index = similarities.argmax()
                            
                            full_text = topic_texts.iloc[top_index]
                            abbreviations = [
                                "Pte. Ltd.",
                                "Co.",
                                "Inc.",
                                "Ltd.",
                                "Corp."
                            ]
                            
                            protected_text = re.sub(r'\b([A-Z]\.)\s*', lambda m: f"INITIAL{ord(m.group(1)[0])}", str(full_text))
                            
                            for i, abbr in enumerate(abbreviations):
                                protected_text = protected_text.replace(abbr, f"ABBR{i}")
                            
                            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected_text)
                            
                            for i, abbr in enumerate(abbreviations):
                                sentences = [s.replace(f"ABBR{i}", abbr) for s in sentences]
                            
                            sentences = [re.sub(r'INITIAL(\d+)', lambda m: chr(int(m.group(1))) + ".", s) for s in sentences]
                            
                            topic_title = sentences[0] if sentences else str(full_text)
                            topic_titles[topic_name] = topic_title
                            
                    except Exception as e:
                        print(f"Error generating title for {topic_name}: {str(e)}")
                        topic_titles[topic_name] = f"Untitled Topic {topic_num}"
                
                topic_data[threshold]['topics'] = list(range(1, current_topic + 1))
                topic_data[threshold]['titles'] = topic_titles
                topic_data[threshold]['counts'] = topic_counts
                
                for topic_name, title in topic_titles.items():
                    df.loc[df[f'Topics_threshold_{threshold}'] == topic_name, f'Topic_Title_threshold_{threshold}'] = title
            
            output_filename = os.path.join(output_dir, "topic_clustering_results.xlsx")
            with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Results', index=False)
                
                # Create separate sheets for each threshold
                for threshold in [threshold1, threshold2, threshold3]:
                    sheet_name = f'Topics_Threshold_{threshold}'
                    pivot_df = pd.DataFrame(columns=['Topic', 'Topic Title', 'Document Count', 'Titles in Topic'])
                    
                    current_row = 0
                    for topic_name in sorted(df[f'Topics_threshold_{threshold}'].unique()):
                        topic_docs = df[df[f'Topics_threshold_{threshold}'] == topic_name]
                        topic_title = topic_docs[f'Topic_Title_threshold_{threshold}'].iloc[0]
                        titles_in_topic = topic_docs['Title'].tolist() if 'Title' in df.columns else []
                        
                        pivot_df.loc[current_row] = [topic_name, topic_title, len(topic_docs), '']
                        current_row += 1
                        
                        for title in titles_in_topic:
                            pivot_df.loc[current_row] = ['', '', '', title]
                            current_row += 1
                    
                    pivot_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    worksheet = writer.sheets[sheet_name]
                    
                    header_fill = PatternFill(start_color='CCCCCC', end_color='CCCCCC', fill_type='solid')
                    border = Border(left=Side(style='thin'), right=Side(style='thin'),
                                  top=Side(style='thin'), bottom=Side(style='thin'))
                    header_font = Font(bold=True)
                    
                    for cell in worksheet[1]:
                        cell.fill = header_fill
                        cell.font = header_font
                        cell.border = border
                        cell.alignment = Alignment(horizontal='center', vertical='center')
                    
                    for row_idx, row in enumerate(worksheet.iter_rows(min_row=2), 2):
                        for cell in row:
                            cell.border = border
                            
                            if cell.column == 1:
                                cell.alignment = Alignment(horizontal='center')
                            elif cell.column == 2:
                                cell.alignment = Alignment(horizontal='left')
                            elif cell.column == 3:
                                cell.alignment = Alignment(horizontal='center')
                            elif cell.column == 4:
                                cell.alignment = Alignment(horizontal='left')
                                if cell.value:
                                    worksheet.row_dimensions[row_idx].outline_level = 1
                    
                    worksheet.column_dimensions['A'].width = 15
                    worksheet.column_dimensions['B'].width = 60
                    worksheet.column_dimensions['C'].width = 15
                    worksheet.column_dimensions['D'].width = 60
                    
                    worksheet.sheet_properties.outlinePr.summaryBelow = False
                    for row in worksheet.row_dimensions:
                        if worksheet.row_dimensions[row].outline_level == 1:
                            worksheet.row_dimensions[row].hidden = True
            
            print(f"\nResults saved to {output_filename}")
            
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
            
            return df
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return None

if __name__ == "__main__":
    analyzer = TextAnalyzer()
    
    input_file = input("Enter the path to your input Excel file: ")
    output_dir = input("Enter the directory where output files should be saved: ")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    text_column = input("\nEnter the column name for topic clustering: ")
    threshold1 = float(input("Enter first similarity threshold (0.0-1.0): "))
    threshold2 = float(input("Enter second similarity threshold (0.0-1.0): "))
    threshold3 = float(input("Enter third similarity threshold (0.0-1.0): "))
    batch_size = int(input("Enter batch size (default 500): ") or 500)
    
    print("\nStarting topic clustering...")
    clustering_results = analyzer.perform_topic_clustering(input_file, text_column, output_dir, threshold1, threshold2, threshold3, batch_size)
    if clustering_results is not None:
        print(f"Topic clustering complete. Results saved to '{os.path.join(output_dir, 'topic_clustering_results.xlsx')}'")
    else:
        print("Topic clustering failed.")
