import gradio as gr
import os
import re
import pandas as pd
import numpy as np
import glob
import huggingface_hub
print("hfh", huggingface_hub.__version__)
from huggingface_hub import hf_hub_download, upload_file, delete_file, snapshot_download, list_repo_files, dataset_info

DATASET_REPO_ID = "AnimaLab/bias-test-gpt-sentences"
DATASET_REPO_URL = f"https://huggingface.co/{DATASET_REPO_ID}"
HF_DATA_DIRNAME = "data"
LOCAL_DATA_DIRNAME = "data"
LOCAL_SAVE_DIRNAME = "save"

ds_write_token = os.environ.get("DS_WRITE_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN")

print("ds_write_token:", ds_write_token!=None)
print("hf_token:", HF_TOKEN!=None)
print("hfh_verssion", huggingface_hub.__version__)

def retrieveAllSaved():
    global DATASET_REPO_ID

    #listing the files - https://huggingface.co/docs/huggingface_hub/v0.8.1/en/package_reference/hf_api
    repo_files = list_repo_files(repo_id=DATASET_REPO_ID, repo_type="dataset")
    #print("Repo files:" + str(repo_files)

    return repo_files

def store_group_sentences(filename: str, df):
  DATA_FILENAME_1 = f"{filename}"
  LOCAL_PATH_FILE = os.path.join(LOCAL_SAVE_DIRNAME, DATA_FILENAME_1)
  DATA_FILE_1 = os.path.join(HF_DATA_DIRNAME, DATA_FILENAME_1)

  print(f"Trying to save to: {DATA_FILE_1}")

  os.makedirs(os.path.dirname(LOCAL_PATH_FILE), exist_ok=True)
  df.to_csv(LOCAL_PATH_FILE, index=False)

  commit_url = upload_file(
    path_or_fileobj=LOCAL_PATH_FILE,
    path_in_repo=DATA_FILE_1,
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
    token=ds_write_token,
  )

  print(commit_url)

def saveSentences(sentences_df):
  for grp_term in list(sentences_df['org_grp_term'].unique()):
    print(f"Retrieving sentences for group: {grp_term}")
    msg, grp_saved_df, filename = getSavedSentences(grp_term)
    print(f"Num for group: {grp_term} -> {grp_saved_df.shape[0]}")
    add_df = sentences_df[sentences_df['org_grp_term'] == grp_term]
    print(f"Adding {add_df.shape[0]} sentences...")
    
    new_grp_df = pd.concat([grp_saved_df, add_df], ignore_index=True)
    new_grp_df = new_grp_df.drop_duplicates(subset = "sentence")

    print(f"Org size: {grp_saved_df.shape[0]}, Mrg size: {new_grp_df.shape[0]}")
    store_group_sentences(filename, new_grp_df)
   

# https://huggingface.co/spaces/elonmuskceo/persistent-data/blob/main/app.py
def get_sentence_csv(file_path: str):
  file_path = os.path.join(HF_DATA_DIRNAME, file_path)
  print(f"File path: {file_path}")
  try:
    hf_hub_download(
       force_download=True, # to get updates of the dataset
       repo_type="dataset",
       repo_id=DATASET_REPO_ID,
       filename=file_path,
       cache_dir=LOCAL_DATA_DIRNAME,
       force_filename=os.path.basename(file_path)
    )
  except Exception as e:
    # file not found
    print(f"file not found, probably: {e}")

  files=glob.glob(f"./{LOCAL_DATA_DIRNAME}/", recursive=True)
  print("Files glob: "+', '.join(files))
  #print("Save file:" + str(os.path.basename(file_path)))
  
  df = pd.read_csv(os.path.join(LOCAL_DATA_DIRNAME, os.path.basename(file_path)), encoding='UTF8')
  
  return df

def getSavedSentences(grp):
    filename = f"{grp.replace(' ','-')}.csv"
    sentence_df = pd.DataFrame()

    try:
        text = f"Loading sentences: {filename}\n"
        sentence_df = get_sentence_csv(filename)

    except Exception as e:
        text = f"Error, no saved generations for {filename}"
        #raise gr.Error(f"Cannot load sentences: {filename}!")

    return text, sentence_df, filename


def deleteBias(filepath: str):
   commit_url = delete_file(
      path_in_repo=filepath,
      repo_id=DATASET_REPO_ID,
      repo_type="dataset",
      token=ds_write_token,
   )

   return f"Deleted {filepath} -> {commit_url}"

def _testSentenceRetrieval(grp_list, att_list, use_paper_sentences):
  test_sentences = []
  print(f"Att list: {att_list}")
  att_list_dash = [t.replace(' ','-') for t in att_list]
  att_list.extend(att_list_dash)
  att_list_nospace = [t.replace(' ','') for t in att_list]
  att_list.extend(att_list_nospace)
  att_list = list(set(att_list))
  print(f"Att list with dash: {att_list}")

  for gi, g_term in enumerate(grp_list):
    _, sentence_df, _ = getSavedSentences(g_term)
    
    # only take from paper & gpt3.5
    print(f"Before filter: {sentence_df.shape[0]}")
    if use_paper_sentences == True:
      if 'type' in list(sentence_df.columns):
        gen_models = ["gpt-3.5", "gpt-3.5-turbo", "gpt-4"] 
        sentence_df = sentence_df.query("type=='paper' and gen_model in @gen_models")
        print(f"After filter: {sentence_df.shape[0]}")
      else:
        sentence_df = pd.DataFrame(columns=["Group term","Attribute term","Test sentence"])

      if sentence_df.shape[0] > 0:
        sentence_df = sentence_df[["Group term","Attribute term","Test sentence"]]
        sel = sentence_df[sentence_df['Attribute term'].isin(att_list)].values
        if len(sel) > 0:
          for gt,at,s in sel:
            test_sentences.append([s,gt.replace("-"," "),at.replace("-"," ")])

    return test_sentences

if __name__ == '__main__':
  print("ds_write_token:", ds_write_token)
  print("hf_token:", HF_TOKEN!=None)
  print("hfh_verssion", huggingface_hub.__version__)

  sentences = _testSentenceRetrieval(["husband"], ["hairdresser", "steel worker"], use_paper_sentences=True)
  print(sentences)

