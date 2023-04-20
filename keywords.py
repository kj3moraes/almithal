import models as md
import pandas as pd

class KeyTakeaways:
    
    def __init__(self, title_element:str):
        self.title_element = []
        self.kw_model = md.load_keyword_model()
        
    def get_keywords(self, text_chunks_lib:dict) -> list:
        kw_dict = dict()
        text_chunk_counter = 0
        
        for key in text_chunks_lib:
            keywords_list = []
            for text_chunk in text_chunks_lib[key]:
                text_chunk_counter += 1
                keywords_list += md.keyword_gen(self.kw_model, text_chunk)
                kw_dict[key] = dict(keywords_list)
        # Display as a dataframe
        kw_df0 = pd.DataFrame.from_dict(kw_dict).reset_index()
        kw_df0.rename(columns={'index': 'keyword'}, inplace=True)
        kw_df = pd.melt(kw_df0, id_vars=['keyword'], var_name='title', value_name='score').dropna()

        kw_column_list = ['keyword', 'score']
        kw_df = kw_df[kw_df['score'] > 0.25][kw_column_list].sort_values(['score'], ascending=False).reset_index().drop(columns='index')

        return kw_df['keyword']