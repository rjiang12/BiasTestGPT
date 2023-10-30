import pandas as pd
import gradio as gr
import hashlib, base64
import openai
from tqdm import tqdm
tqdm().pandas()

# querying OpenAI for generation
import openAI_manager as oai_mgr 
#import initOpenAI, examples_to_prompt, genChatGPT, generateTestSentences

# bias testing manager
import mgr_bias_scoring as bt_mgr
import mgr_sentences as smgr

# error messages
from error_messages import *

G_CORE_BIAS_NAME = None

# hashing
def getHashForString(text):
  d=hashlib.md5(bytes(text, encoding='utf-8')).digest()
  d=base64.urlsafe_b64encode(d)

  return d.decode('utf-8')

def getBiasName(gr1_lst, gr2_lst, att1_lst, att2_lst):
    global G_CORE_BIAS_NAME

    bias_name = G_CORE_BIAS_NAME
    if bias_name == None:
        full_spec = ''.join(gr1_lst)+''.join(gr2_lst)+''.join(att1_lst)+''.join(att2_lst)
        hash = getHashForString(full_spec)
        bias_name = f"{gr1_lst[0].replace(' ','-')}_{gr2_lst[0].replace(' ','-')}__{att1_lst[0].replace(' ','-')}_{att2_lst[0].replace(' ','-')}_{hash}"
  
    return bias_name

def _generateOnline(bias_spec, progress, key, num2gen, isSaving=False):
    test_sentences = []
    gen_err_msg = None
    genAttrCounts = {}
    print(f"Bias spec dict: {bias_spec}")
    g1, g2, a1, a2 = bt_mgr.get_words(bias_spec)
    print(f"A1: {a1}")
    print(f"A2: {a2}")

    if "custom_counts" in bias_spec:
        print("Bias spec is custom !!")
        genAttrCounts = bias_spec['custom_counts'][0]
        for a,c in bias_spec['custom_counts'][1].items():
            genAttrCounts[a] = c
    else:
        print("Bias spec is standard !!")
        genAttrCounts = {a:num2gen for a in a1+a2}

    # Initiate with key
    try:
        models = oai_mgr.initOpenAI(key)
        model_names = [m['id'] for m in models['data']]
        print(f"Model names: {model_names}")
    except openai.error.AuthenticationError as err:
        #raise gr.Error(OPENAI_INIT_ERROR.replace("<ERR>", str(err)))
        gen_err_msg = OPENAI_INIT_ERROR.replace("<ERR>", str(err))
    
    if gen_err_msg != None:
        return [], gen_err_msg
    else:
        if "gpt-3.5-turbo" in model_names:
            print("Access to ChatGPT")
        if "gpt-4" in model_names:
            print("Access to GPT-4")

        model_name = "gpt-3.5-turbo" #"gpt-4"

        # Generate one example
        #gen = genChatGPT(model_name, ["man","math"], 2, 5, 
        #            [{"Keywords": ["sky","blue"], "Sentence": "the sky is blue"}
        #            ], 
        #            temperature=0.8)
        #print(f"Test gen: {gen}")

        # Generate all test sentences
        
        #gens = oai_mgr.generateTestSentences(model_name, g1+g2, a1+a2, num2gen, progress)
        gens = oai_mgr.generateTestSentencesCustom(model_name, g1, g2, a1+a2, genAttrCounts, bias_spec, progress)
        print("--GENS--")
        print(gens)
        if len(gens) == 0:
            print("No sentences generated, returning")
            return [], gen_err_msg

        for org_gt, at, s, gt1, gt2 in gens:
            test_sentences.append([s,org_gt,at,gt1,gt2])

        # save the generations immediately
        print("Making save dataframe...")
        save_df = pd.DataFrame(test_sentences, columns=["Sentence",'org_grp_term', 
                                                        "Attribute term", "Group term 1", 
                                                        "Group term 2"])

        ## make the templates to save
        # 1. bias specification
        print(f"Bias spec dict: {bias_spec}")

        # generate laternative sentence
        print(f"Columns before alternative sentence: {list(save_df.columns)}")
        save_df['Alternative Sentence'] = save_df.progress_apply(oai_mgr.chatgpt_sentence_alternative, axis=1, model_name=model_name)
        print(f"Columns after alternative sentence: {list(save_df.columns)}")

        # 2. convert to templates
        save_df['Template'] = save_df.progress_apply(bt_mgr.sentence_to_template_df, axis=1)
        print("Convert generated sentences to templates...")
        save_df[['Alternative Template','grp_refs']] = save_df.progress_apply(bt_mgr.ref_terms_sentence_to_template, axis=1)
        print(f"Columns with templates: {list(save_df.columns)}")

        # 3. convert to pairs
        print("Convert generated sentences to ordered pairs...")
        test_pairs_df = bt_mgr.convert2pairsFromDF(bias_spec, save_df)
        print(f"Test pairs cols: {list(test_pairs_df.columns)}")

        bias_name = getBiasName(g1, g2, a1, a2)

        save_df = save_df.rename(columns={"Sentence":'sentence',
                                          "Alternative Sentence":"alt_sentence",
                                "Attribute term": 'att_term',
                                "Template":"template",
                                "Alternative Template": "alt_template",
                                "Group term 1": "grp_term1",
                                "Group term 2": "grp_term2"})
        
        save_df['label_1'] = test_pairs_df['label_1']
        save_df['label_2'] = test_pairs_df['label_2']
        save_df['bias_spec'] = bias_name
        save_df['type'] = 'tool'
        save_df['gen_model'] = model_name

        col_order = ["sentence", "alt_sentence", "org_grp_term", "att_term", "template", 
                     "alt_template", "grp_term1", "grp_term2", "grp_refs", "label_1", "label_2",
                     "bias_spec", "type", "gen_model"]
        save_df = save_df[col_order]

        print(f"Save cols prep: {list(save_df.columns)}")

        if isSaving == True:
            print(f"Saving: {save_df.head(1)}")
            smgr.saveSentences(save_df) #[["Group term","Attribute term","Test sentence"]])

        num_sentences = len(test_sentences)
        print(f"Returned num sentences: {num_sentences}")

        # list for Gradio dataframe
        ret_df = [list(r.values) for i, r in save_df[['sentence', 'alt_sentence', 'grp_term1', 'grp_term2', "att_term"]].iterrows()]
        print(ret_df)

        return ret_df, gen_err_msg

def _getSavedSentences(bias_spec, progress, use_paper_sentences):
    test_sentences = []

    print(f"Bias spec dict: {bias_spec}")

    g1, g2, a1, a2 = bt_mgr.get_words(bias_spec)
    for gi, g_term in enumerate(g1+g2):
        att_list = a1+a2
        grp_list = g1+g2
        # match "-" and no space
        att_list_dash = [t.replace(' ','-') for t in att_list]
        att_list.extend(att_list_dash)
        att_list_nospace = [t.replace(' ','') for t in att_list]
        att_list.extend(att_list_nospace)
        att_list = list(set(att_list))

        progress(gi/len(g1+g2), desc=f"{g_term}")

        _, sentence_df, _ = smgr.getSavedSentences(g_term)
        # only take from paper & gpt3.5
        flt_gen_models = ["gpt-3.5","gpt-3.5-turbo","gpt-4"]
        print(f"Before filter: {sentence_df.shape[0]}")
        if use_paper_sentences == True:
            if 'type' in list(sentence_df.columns):
                sentence_df = sentence_df.query("type=='paper' and gen_model in @flt_gen_models")
                print(f"After filter: {sentence_df.shape[0]}")
        else:
            if 'type' in list(sentence_df.columns):
                # only use GPT-3.5 generations for now - todo: add settings option for this
                sentence_df = sentence_df.query("gen_model in @flt_gen_models")
                print(f"After filter: {sentence_df.shape[0]}")

        if sentence_df.shape[0] > 0:
            sentence_df = sentence_df[['grp_term1','grp_term2','att_term','sentence','alt_sentence']]
            sentence_df = sentence_df.rename(columns={'grp_term1': "Group term 1",
                                                      'grp_term2': "Group term 2",
                                                        "att_term": "Attribute term",
                                                        "sentence": "Sentence",
                                                        "alt_sentence": "Alt Sentence"})

            sel = sentence_df[(sentence_df['Attribute term'].isin(att_list)) & \
                              ((sentence_df['Group term 1'].isin(grp_list)) & (sentence_df['Group term 2'].isin(grp_list))) ].values
            if len(sel) > 0:
                for gt1,gt2,at,s,a_s in sel:
                    #if at == "speech-language-pathologist":
                    #    print(f"Special case: {at}")
                    #    at == "speech-language pathologist" # legacy, special case
                    #else:
                    #at = at #.replace("-"," ")
                    #gt = gt #.replace("-"," ")

                    test_sentences.append([s,a_s,gt1,gt2,at])
        else:
            print("Test sentences empty!")
            #raise gr.Error(NO_SENTENCES_ERROR)

    return test_sentences