import gradio as gr
import pandas as pd
import numpy as np
import string
import re
import json
import random
import torch
import hashlib, base64
from tqdm import tqdm
from gradio.themes.base import Base
import openai

# bloomber vis
import bloomberg_vis as bv

# error messages
from error_messages import *

tqdm().pandas()

# bias testing manager
import mgr_bias_scoring as bt_mgr

# managers for sentences and biases
import mgr_requests as rq_mgr
from mgr_requests import G_CORE_BIAS_NAME
import mgr_biases as bmgr

# cookie manager
#import mgr_cookies as cookie_mgr

use_paper_sentences = False
G_TEST_SENTENCES = []
G_NUM_SENTENCES = 0
G_MISSING_SPEC = []

def getTermsFromGUI(group1, group2, att1, att2):
    bias_spec = {
      "social_groups": {
        "group 1": [t.strip(" ") for t in group1.split(",") if len(t.strip(' '))>0], 
        "group 2": [t.strip(" ") for t in group2.split(",") if len(t.strip(' '))>0]},
      "attributes": {
        "attribute 1": [t.strip(" ") for t in att1.split(",") if len(t.strip(' '))>0], 
        "attribute 2": [t.strip(" ") for t in att2.split(",") if len(t.strip(' '))>0]}
    }
    return bias_spec

# Select from example datasets
def prefillBiasSpec(evt: gr.SelectData):
    global use_paper_sentences, G_MISSING_SPEC, G_CORE_BIAS_NAME

    G_MISSING_SPEC = []
    G_CORE_BIAS_NAME = evt.value
    print(f"Setting core bias name to: {G_CORE_BIAS_NAME}")

    print(f"Selected {evt.value} at {evt.index} from {evt.target}")
    #bias_filename = f"{evt.value[1]}.json"
    bias_filename = f"{bmgr.bias2tag[evt.value]}.json"
    print(f"Filename: {bias_filename}")

    isCustom = bmgr.isCustomBias(bias_filename)
    if isCustom:
        print(f"Custom bias specification: {bias_filename}")
        bias_spec = bmgr.loadCustomBiasSpec(bias_filename)
    else:
        print(f"Core bias specification: {bias_filename}")
        bias_spec = bmgr.loadPredefinedBiasSpec(bias_filename)

    grp1_terms, grp2_terms = bmgr.getSocialGroupTerms(bias_spec)
    att1_terms, att2_terms = bmgr.getAttributeTerms(bias_spec)

    print(f"Grp 1: {grp1_terms}")
    print(f"Grp 2: {grp2_terms}")

    print(f"Att 1: {att1_terms}")
    print(f"Att 2: {att2_terms}")

    #use_paper_sentences = True

    return (', '.join(grp1_terms[0:50]), ', '.join(grp2_terms[0:50]), ', '.join(att1_terms[0:50]), ', '.join(att2_terms[0:50]),
            gr.update(interactive=False, visible=False))

def updateErrorMsg(isError, text):
    return gr.Markdown.update(visible=isError, value=text)

def countBiasCustomSpec(bias_spec):
    if (bias_spec) == 0:
        return 0
    elif 'custom_counts' in bias_spec:
        rq_count_1 = sum([v for v in bias_spec['custom_counts' ][0].values()])
        rq_count_2 = sum([v for v in bias_spec['custom_counts' ][1].values()])

        return rq_count_1+rq_count_2
    else:
        return 0

def generateSentences(gr1, gr2, att1, att2, openai_key, num_sent2gen, progress=gr.Progress()):
    global use_paper_sentences, G_NUM_SENTENCES, G_MISSING_SPEC, G_TEST_SENTENCES
    print(f"GENERATE SENTENCES CLICKED!, requested sentence per attribute number: {num_sent2gen}")

    # No error messages by default
    err_update = updateErrorMsg(False, "")
    bias_test_label = "Test Model Using Imbalanced Sentences"
    
    # There are no sentences available at all
    if len(G_TEST_SENTENCES) == 0:
        bias_gen_states = [True, False]
        online_gen_visible = True
        test_model_visible = False
    else:
        bias_gen_states = [True, True]
        online_gen_visible = True
        test_model_visible = True
    info_msg_update = gr.Markdown.update(visible=False, value="")

    test_sentences = []
    bias_spec = getTermsFromGUI(gr1, gr2, att1, att2)
    g1, g2, a1, a2 = bt_mgr.get_words(bias_spec)
    total_att_terms = len(a1)+len(a2)
    all_terms_len = len(g1)+len(g2)+len(a1)+len(a2)
    print(f"Length of all the terms: {all_terms_len}")
    if all_terms_len == 0:
        print("No terms entered!")
        err_update = updateErrorMsg(True, NO_TERMS_ENTERED_ERROR) 
        #raise gr.Error(NO_TERMS_ENTERED_ERROR)
    else:
        if len(openai_key) == 0:
            print("Empty OpenAI key!!!")
            err_update = updateErrorMsg(True, OPENAI_KEY_EMPTY) 
        elif len(openai_key) < 10:
            print("Wrong length OpenAI key!!!")
            err_update = updateErrorMsg(True, OPENAI_KEY_WRONG) 
        else:
            progress(0, desc="ChatGPT generation...")
            print(f"Using Online Generator LLM...")

            print(f"Is custom spec? {countBiasCustomSpec(G_MISSING_SPEC)}")
            print(f"Custom spec: {G_MISSING_SPEC}")
            use_bias_spec = G_MISSING_SPEC if countBiasCustomSpec(G_MISSING_SPEC)>0 else bias_spec
            test_sentences, gen_err_msg = rq_mgr._generateOnline(use_bias_spec, progress, openai_key, num_sent2gen, isSaving=False)

            #print(f"Test sentences: {test_sentences}")
            num_sentences = len(test_sentences)
            print(f"Returned num sentences: {num_sentences}")

            G_NUM_SENTENCES = len(G_TEST_SENTENCES) + num_sentences
            if num_sentences == 0 and len(G_TEST_SENTENCES) == 0:
                print("Test sentences empty!")
                #raise gr.Error(NO_SENTENCES_ERROR)  
                
                # Some error returned from OpenAI generator
                if gen_err_msg != None:
                    err_update = updateErrorMsg(True, gen_err_msg)         
                # No sentences returned, but no specific error
                else:
                    err_update = updateErrorMsg(True, NO_GEN_SENTENCES_ERROR)
            elif num_sentences == 0 and len(G_TEST_SENTENCES) > 0:
                print(f"Has some retrieved sentences {G_TEST_SENTENCES}, but no sentnces generated {num_sentences}!")
                #raise gr.Error(NO_SENTENCES_ERROR)  
                
                # Some error returned from OpenAI generator
                if gen_err_msg != None:
                    err_update = updateErrorMsg(True, gen_err_msg)         
                # No sentences returned, but no specific error
                else:
                    err_update = updateErrorMsg(True, NO_GEN_SENTENCES_ERROR)
                 # has all sentences, can bias test
                bias_gen_states = [True, True]
                
            else:
                print("Combining generated and existing...")
                print(f"Existing sentences: {len(G_TEST_SENTENCES)}")
                print(f"Generated: {len(test_sentences)}")
                G_TEST_SENTENCES = G_TEST_SENTENCES + test_sentences
                print(f"Combined: {len(G_TEST_SENTENCES)}")
                # has all sentences, can bias test
                bias_gen_states = [False, True]
                online_gen_visible = False
                test_model_visible = True # show choise of tested model and the sentences
                info_msg, att1_missing, att2_missing, total_missing, c_bias_spec = _genSentenceCoverMsg(G_TEST_SENTENCES, total_att_terms, bias_spec, isGen=True)
          
                info_msg_update = gr.Markdown.update(visible=True, value=info_msg)
                bias_test_label = "Test Model For Social Bias"

                #cookie_mgr.saveOpenAIKey(openai_key)

    print(f"Online gen visible: {not err_update['visible']}")
    return (err_update, # err message if any
        info_msg_update, # infor message about the number of sentences and coverage
        gr.Row.update(visible=online_gen_visible),    # online gen row
        #gr.Slider.update(minimum=8, maximum=24, value=4), # slider generation
        gr.Row.update(visible=test_model_visible), # tested model row 
        #gr.Dropdown.update(visible=test_model_visible), # tested model selection dropdown
        gr.Accordion.update(visible=test_model_visible, label=f"Test sentences ({len(G_TEST_SENTENCES)})"), # accordion
        gr.update(visible=True), # Row sentences
        gr.DataFrame.update(value=G_TEST_SENTENCES), #DataFrame test sentences
        gr.update(visible=bias_gen_states[0]), # gen btn
        gr.update(visible=bias_gen_states[1], value=bias_test_label)  # bias btn
)

# Interaction with top tabs
def moveStep1():
    variants = ["primary","secondary","secondary"]
    #inter = [True, False, False]
    tabs = [True, False, False]

    return (gr.update(variant=variants[0]),
            gr.update(variant=variants[1]),
            gr.update(variant=variants[2]),
            gr.update(visible=tabs[0]),
            gr.update(visible=tabs[1]),
            gr.update(visible=tabs[2]))

# Interaction with top tabs
def moveStep1_clear():
    variants = ["primary","secondary","secondary"]
    #inter = [True, False, False]
    tabs = [True, False, False]

    return (gr.update(variant=variants[0]),
            gr.update(variant=variants[1]),
            gr.update(variant=variants[2]),
            gr.update(visible=tabs[0]),
            gr.update(visible=tabs[1]),
            gr.update(visible=tabs[2]),
            gr.Textbox.update(value=""),
            gr.Textbox.update(value=""),
            gr.Textbox.update(value=""),
            gr.Textbox.update(value=""))

def moveStep2():
    variants = ["secondary","primary","secondary"]
    #inter = [True, True, False]
    tabs = [False, True, False]

    return (gr.update(variant=variants[0]),
            gr.update(variant=variants[1]),
            gr.update(variant=variants[2]),
            gr.update(visible=tabs[0]),
            gr.update(visible=tabs[1]),
            gr.update(visible=tabs[2]),
            gr.Checkbox.update(value=False))

def moveStep3():
    variants = ["secondary","secondary","primary"]
    #inter = [True, True, False]
    tabs = [False, False, True]

    return (gr.update(variant=variants[0]),
            gr.update(variant=variants[1]),
            gr.update(variant=variants[2]),
            gr.update(visible=tabs[0]),
            gr.update(visible=tabs[1]),
            gr.update(visible=tabs[2]))

def _genSentenceCoverMsg(test_sentences, total_att_terms, bias_spec, isGen=False):
    att_cover_dict = {}
    print(f"In Coverage: {test_sentences[0:2]}")
    for sent,alt_sent,gt1,gt2,att in test_sentences:
        num = att_cover_dict.get(att, 0)
        att_cover_dict[att] = num+1
    att_by_count = dict(sorted(att_cover_dict.items(), key=lambda item: item[1]))
    num_covered_atts = len(list(att_by_count.keys()))
    lest_covered_att = list(att_by_count.keys())[0]
    least_covered_count = att_by_count[lest_covered_att]

    test_sentences_df = pd.DataFrame(test_sentences, columns=['sentence', 'alt_sentence', "grp_term1", "grp_term2", "att_term"])

    # missing sentences for attributes
    att1_missing, att2_missing = bt_mgr.genMissingAttribBiasSpec(bias_spec, test_sentences_df)
    print(f"Att 1 missing: {att1_missing}")
    print(f"Att 2 missing: {att2_missing}")

    # missing pairs spec
    bt_mgr.genMissingPairsSpec(bias_spec, test_sentences_df)



    att1_missing_num = sum([v for k, v in att1_missing.items()])
    att2_missing_num = sum([v for k, v in att2_missing.items()])
    total_missing = att1_missing_num + att2_missing_num

    print(f"Total missing: {total_missing}")
    missing_info = f"Missing {total_missing} sentences to balance attributes <bt /> "

    source_msg = "Found" if isGen==False else "Generated"
    if num_covered_atts >= total_att_terms:
        if total_missing > 0:
            info_msg = f"**{source_msg} {len(test_sentences)} sentences covering all bias specification attributes, but some attributes are underepresented. Generating additional {total_missing} sentences is suggested.**"
        else:
            info_msg = f"**{source_msg} {len(test_sentences)} sentences covering all bias specification attributes. Please select model to test.**"
    else:
        info_msg = f"**{source_msg} {len(test_sentences)} sentences covering {num_covered_atts} of {total_att_terms} attributes. Please select model to test.**"

    #info_msg = missing_info + info_msg
    bias_spec['custom_counts'] = [att1_missing, att2_missing]

    return info_msg, att1_missing, att2_missing, total_missing, bias_spec

def retrieveSentences(gr1, gr2, att1, att2, progress=gr.Progress()):
    global use_paper_sentences, G_NUM_SENTENCES, G_MISSING_SPEC, G_TEST_SENTENCES

    print("RETRIEVE SENTENCES CLICKED!")
    G_MISSING_SPEC = []
    variants = ["secondary","primary","secondary"]
    inter = [True, True, False]
    tabs = [True, False]
    bias_gen_states = [True, False]
    bias_gen_label = "Generate New Sentences"
    bias_test_label = "Test Model for Social Bias"
    num2gen_update = gr.update(visible=True) #update the number of new sentences to generate
    prog_vis = [True]
    err_update = updateErrorMsg(False, "") 
    info_msg_update = gr.Markdown.update(visible=False, value="")
    openai_gen_row_update = gr.Row.update(visible=True)
    tested_model_dropdown_update = gr.Dropdown.update(visible=False)
    tested_model_row_update = gr.Row.update(visible=False)
    # additinal sentences disabled by default
    gen_additional_sentence_checkbox_update = gr.Checkbox.update(visible=False)

    test_sentences = []
    bias_spec = getTermsFromGUI(gr1, gr2, att1, att2)
    g1, g2, a1, a2 = bt_mgr.get_words(bias_spec)
    total_att_terms = len(a1)+len(a2)
    all_terms_len = len(g1)+len(g2)+len(a1)+len(a2)
    print(f"Length of all the terms: {all_terms_len}")
    if all_terms_len == 0:
        print("No terms entered!")
        err_update = updateErrorMsg(True, NO_TERMS_ENTERED_ERROR) 
        variants = ["primary","secondary","secondary"]
        inter = [True, False, False]
        tabs = [True, False]
        prog_vis = [False]

        #raise gr.Error(NO_TERMS_ENTERED_ERROR)
    else:
        tabs = [False, True]
        progress(0, desc="Fetching saved sentences...")
        test_sentences = rq_mgr._getSavedSentences(bias_spec, progress, use_paper_sentences)

        #err_update, _, test_sentences = generateSentences(gr1, gr2, att1, att2, progress)
        print(f"Type: {type(test_sentences)}")
        num_sentences = len(test_sentences)
        print(f"Returned num sentences: {num_sentences}")

        err_update = updateErrorMsg(False, "")
        G_NUM_SENTENCES = num_sentences
        G_TEST_SENTENCES = test_sentences
        if G_NUM_SENTENCES == 0:
            print("Test sentences empty!")
            #raise gr.Error(NO_SENTENCES_ERROR)  
            err_update = updateErrorMsg(True, NO_SENTENCES_ERROR) 

        if len(test_sentences) > 0:
            info_msg, att1_missing, att2_missing, total_missing, c_bias_spec = _genSentenceCoverMsg(test_sentences, total_att_terms, bias_spec)
            G_MISSING_SPEC = c_bias_spec
            print(f"Saving global custom bias specification: {G_MISSING_SPEC}")

            info_msg_update = gr.Markdown.update(visible=True, value=info_msg)
            num2gen_update = gr.update(visible=False)
            bias_gen_label = f"Generate Additional {total_missing} Sentences"

            if total_missing == 0:
                print(f"Got {len(test_sentences)}, allowing bias test...")
                #print(test_sentences)
                bias_gen_states = [False, True]
                openai_gen_row_update = gr.Row.update(visible=False)
                tested_model_dropdown_update = gr.Dropdown.update(visible=True)
                tested_model_row_update = gr.Row.update(visible=True)

                # still give the option to generate more sentences
                gen_additional_sentence_checkbox_update = gr.Checkbox.update(visible=True)

            else:
                bias_test_label = "Test Model Using Imbalanced Sentences"
                bias_gen_states = [True, True]
                tested_model_dropdown_update = gr.Dropdown.update(visible=True)
                tested_model_row_update = gr.Row.update(visible=True)

    return (err_update, # error message
            openai_gen_row_update, # OpenAI generation
            gen_additional_sentence_checkbox_update, # optional generate additional sentences
            num2gen_update, # Number of sentences to genrate 
            tested_model_row_update, #Tested Model Row
            #tested_model_dropdown_update, # Tested Model Dropdown
            info_msg_update, # sentences retrieved info update
            gr.update(visible=prog_vis), # progress bar top
            gr.update(variant=variants[0], interactive=inter[0]), # breadcrumb btn1
            gr.update(variant=variants[1], interactive=inter[1]), # breadcrumb btn2
            gr.update(variant=variants[2], interactive=inter[2]), # breadcrumb btn3
            gr.update(visible=tabs[0]), # tab 1
            gr.update(visible=tabs[1]), # tab 2
            gr.Accordion.update(visible=bias_gen_states[1], label=f"Test sentences ({len(test_sentences)})"), # accordion
            gr.update(visible=True), # Row sentences
            gr.DataFrame.update(value=test_sentences), #DataFrame test sentences
            gr.Button.update(visible=bias_gen_states[0], value=bias_gen_label), # gen btn
            gr.Button.update(visible=bias_gen_states[1], value=bias_test_label), # bias test btn
            gr.update(value=', '.join(g1)), # gr1_fixed
            gr.update(value=', '.join(g2)), # gr2_fixed
            gr.update(value=', '.join(a1)), # att1_fixed
            gr.update(value=', '.join(a2))  # att2_fixed
        )

def startBiasTest(test_sentences_df, gr1, gr2, att1, att2, model_name, progress=gr.Progress()):
    global G_NUM_SENTENCES

    variants = ["secondary","secondary","primary"]
    inter = [True, True, True]
    tabs = [False, False, True]
    err_update = updateErrorMsg(False, "") 

    if test_sentences_df.shape[0] == 0:
      G_NUM_SENTENCES = 0
      #raise gr.Error(NO_SENTENCES_ERROR)
      err_update = updateErrorMsg(True, NO_SENTENCES_ERROR) 

    
    progress(0, desc="Starting social bias testing...")
    
    #print(f"Type: {type(test_sentences_df)}")
    #print(f"Data: {test_sentences_df}")

    # bloomberg vis
    att_freqs = {}
    for att in test_sentences_df["Attribute term"].tolist():
        #if att == "speech-language-pathologist" or att == "speech-language pathologist" or att == "speech language pathologist":
        #    print(f"Special case in bloomberg: {att}")
        #    att = "speech-language pathologist"
        
        if att in att_freqs:
            att_freqs[att] += 1
        else:
            att_freqs[att] = 1

    #print(f"att_freqs: {att_freqs}")

    # 1. bias specification
    bias_spec = getTermsFromGUI(gr1, gr2, att1, att2)
    #print(f"Bias spec dict: {bias_spec}")
    g1, g2, a1, a2 = bt_mgr.get_words(bias_spec)

    # bloomberg vis
    attributes_g1 = a1 #list(set(a1 + [a.replace(' ','-') for a in a1])) #bias_spec['attributes']['attribute 1']
    attributes_g2 = a2 #list(set(a2 + [a.replace(' ','-') for a in a2])) #bias_spec['attributes']['attribute 2']

    #print(f"Attributes 1: {attributes_g1}")
    #print(f"Attributes 2: {attributes_g2}")

    # 2. convert to templates
    #test_sentences_df['Template'] = test_sentences_df.apply(bt_mgr.sentence_to_template_df, axis=1)
    test_sentences_df[['Template','grp_refs']] = test_sentences_df.progress_apply(bt_mgr.ref_terms_sentence_to_template, axis=1)
    print(f"Columns with templates: {list(test_sentences_df.columns)}")
    print(test_sentences_df[['Group term 1', 'Group term 2', 'Sentence', 'Alternative Sentence']])

    # 3. convert to pairs
    test_pairs_df = bt_mgr.convert2pairsFromDF(bias_spec, test_sentences_df)
    print(f"Columns for test pairs: {list(test_pairs_df.columns)}")
    print(test_pairs_df[['grp_term_1', 'grp_term_2', 'sentence', 'alt_sentence']])


    progress(0.05, desc=f"Loading model {model_name}...")
    # 4. get the per sentence bias scores
    print(f"Test model name: {model_name}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tested_model, tested_tokenizer = bt_mgr._getModelSafe(model_name, device)
    if tested_model == None:
        print("Tested model is empty!!!!")
        err_update = updateErrorMsg(True, MODEL_NOT_LOADED_ERROR) 

    #print(f"Mask token id: {tested_toknizer.mask_token_id}")

    # sanity check bias test
    bt_mgr.testModelProbability(model_name, tested_model, tested_tokenizer, device)

    # testing actual sentences
    test_score_df, bias_stats_dict = bt_mgr.testBiasOnPairs(test_pairs_df, bias_spec, model_name, tested_model, tested_tokenizer, device, progress)
    print(f"Test scores: {test_score_df.head(3)}")
    num_sentences = test_sentences_df.shape[0] #score_templates_df.shape[0]

    model_bias_dict = {}
    tested_model = bias_stats_dict['tested_model']
    #model_bias_dict[bias_stats_dict['tested_model']] = bias_stats_dict['model_bias']
    model_bias_dict[f'Stereotype Score on {tested_model.upper()} using {num_sentences} sentences'] = bias_stats_dict['model_bias']
    
    per_attrib_bias = bias_stats_dict['per_attribute']
    #print(f"Per attribute bias:", per_attrib_bias)

    # bias score
    #test_pairs_df['bias_score'] = 0
    test_pairs_df.loc[test_pairs_df['stereotyped'] == 1, 'bias_score'] = test_pairs_df['top_logit']-test_pairs_df['bottom_logit']
    test_pairs_df.loc[test_pairs_df['stereotyped'] == 0, 'bias_score'] = test_pairs_df['bottom_logit']-test_pairs_df['top_logit']

    test_pairs_df['stereotyped_b'] = "Unknown"
    test_pairs_df.loc[test_pairs_df['stereotyped'] == 1, 'stereotyped_b'] = "yes"
    test_pairs_df.loc[test_pairs_df['stereotyped'] == 0, 'stereotyped_b'] = "no"

    # Order group terms such that most probable is first
    def orderGroups(row):
        group_order = "None/None"
        sentence_order = ["none","none"]
        new_grp_refs = [] #list(row['grp_refs'])
        for grp_pair in list(row['grp_refs']):
            new_grp_refs.append(("R1","R2"))
        #print(f"Grp refs: {new_grp_refs}")
        if row['stereotyped'] == 1:
            if row["label_1"] == "stereotype":
                group_order = row['grp_term_1']+"/"+row['grp_term_2']
                sentence_order = [row['sentence'], row['alt_sentence']]
                new_grp_refs = []
                for grp_pair in list(row['grp_refs']):
                    new_grp_refs.append((grp_pair[0], grp_pair[1]))
            else:
                group_order = row['grp_term_2']+"/"+row['grp_term_1']
                sentence_order = [row['alt_sentence'], row['sentence']]
                new_grp_refs = []
                for grp_pair in list(row['grp_refs']):
                    new_grp_refs.append((grp_pair[1], grp_pair[0]))
        else:
            if row["label_1"] == "stereotype":
                group_order = row['grp_term_2']+"/"+row['grp_term_1']
                sentence_order = [row['alt_sentence'], row['sentence']]
                new_grp_refs = []
                for grp_pair in list(row['grp_refs']):
                    new_grp_refs.append((grp_pair[1], grp_pair[0]))
            else:
                group_order = row['grp_term_1']+"/"+row['grp_term_2']
                sentence_order = [row['sentence'], row['alt_sentence']]
                new_grp_refs = []
                for grp_pair in list(row['grp_refs']):
                    new_grp_refs.append((grp_pair[0], grp_pair[1]))
        
        return pd.Series([group_order, sentence_order[0], sentence_order[1], new_grp_refs])

    test_pairs_df[['groups_rel','sentence', 'alt_sentence', 'grp_refs']] = test_pairs_df.progress_apply(orderGroups, axis=1)
    #test_pairs_df['groups_rel'] = test_pairs_df['grp_term_1']+"/"+test_pairs_df['grp_term_2']

    # construct display dataframe
    score_templates_df = test_pairs_df[['att_term','template','sentence','alt_sentence']].copy()
    score_templates_df['Groups'] = test_pairs_df['groups_rel']
    #score_templates_df['Bias Score'] = np.round(test_pairs_df['bias_score'],2)
    score_templates_df['Stereotyped'] = test_pairs_df['stereotyped_b']

    score_templates_df = score_templates_df.rename(columns = {'att_term': "Attribute",
                                                               "template": "Template",
                                                               "sentence": "Sentence",
                                                               "alt_sentence": "Alternative"})
    #'Bias Score'
    score_templates_df = score_templates_df[['Stereotyped','Attribute','Groups','Sentence',"Alternative"]]

    # bloomberg vis
    attrib_by_score = dict(sorted(per_attrib_bias.items(), key=lambda item: item[1], reverse=True))
    #print(f"Attrib by score:", attrib_by_score)

    per_attrib_bias_HTML_stereo = ""
    num_atts = 0
    for att, score in attrib_by_score.items():
        if att in attributes_g1:
            #print(f"Attribute 1: {att}")
            #per_attrib_bias_HTML_stereo += bv.att_bloombergViz(att, score, att_freqs[att])
            #num_atts += 1
            #if num_atts >= 8:
            #    break

            per_attrib_bias_HTML_stereo += bv.att_bloombergViz(att, score, att_freqs[att], test_pairs_df, False, False)
            num_atts += 1
            #if num_atts >= 8:
            #    break

    per_attrib_bias_HTML_antistereo = ""
    num_atts = 0
    for att, score in attrib_by_score.items():
        if att in attributes_g2:
            #print(f"Attribute 2: {att}")
            #per_attrib_bias_HTML_antistereo += bv.att_bloombergViz(att, score, att_freqs[att], True)
            #num_atts += 1
            #if num_atts >= 8:
            #    break

            per_attrib_bias_HTML_antistereo += bv.att_bloombergViz(att, score, att_freqs[att], test_pairs_df, True, True)
            num_atts += 1
            #if num_atts >= 8:
            #    break

    interpret_msg = bt_mgr._constructInterpretationMsg(bias_spec, num_sentences, 
                                                       model_name, bias_stats_dict, per_attrib_bias,
                                                       score_templates_df
                                                       )
    
    saveBiasTestResult(test_sentences_df, gr1, gr2, att1, att2, model_name)

    return (err_update, # error message
            gr.Markdown.update(visible=True), # bar progress
            gr.Button.update(variant=variants[0], interactive=inter[0]), # top breadcrumb button 1
            gr.Button.update(variant=variants[1], interactive=inter[1]), # top breadcrumb button 2
            gr.Button.update(variant=variants[2], interactive=inter[2]), # top breadcrumb button 3
            gr.update(visible=tabs[0]), # content tab/column 1
            gr.update(visible=tabs[1]), # content tab/column 2
            gr.update(visible=tabs[2]), # content tab/column 3
            model_bias_dict, # per model bias score
            gr.update(value=per_attrib_bias_HTML_stereo), # per attribute bias score stereotyped
            gr.update(value=per_attrib_bias_HTML_antistereo), # per attribute bias score antistereotyped
            gr.update(value=score_templates_df, visible=True), # Pairs with scores
            gr.update(value=interpret_msg, visible=True), # Interpretation message
            gr.update(value=', '.join(g1)), # gr1_fixed
            gr.update(value=', '.join(g2)), # gr2_fixed
            gr.update(value=', '.join(a1)), # att1_fixed
            gr.update(value=', '.join(a2))  # att2_fixed
            )

# Loading the Interface first time
def loadInterface():
    print("Loading the interface...")
    #open_ai_key = cookie_mgr.loadOpenAIKey()

    #return gr.Textbox.update(value=open_ai_key)

# Selecting an attribute label in the label component
def selectAttributeLabel(evt: gr.SelectData):
    print(f"Selected {evt.value} at {evt.index} from {evt.target}")
    object_methods = [method_name for method_name in dir(evt)
                  if callable(getattr(evt, method_name))]
    
    print("Attributes:")
    for att in dir(evt):
        print (att, getattr(evt,att))
    
    print(f"Methods: {object_methods}")

    return ()

# Editing a sentence in DataFrame
def editSentence(test_sentences, evt: gr.EventData):
    print(f"Edit Sentence: {evt}")
    #print("--BEFORE---")
    #print(test_sentences[0:10])
    #print("--AFTER--")
    #print(f"Data: {evt._data['data'][0:10]}")
    # print("Attributes:")
    # for att in dir(evt):
    #     print (att, getattr(evt,att))

    # object_methods = [method_name for method_name in dir(evt)
    #               if callable(getattr(evt, method_name))]
    
    # print(f"Methods: {object_methods}")

# exports dataframe as CSV
def export_csv(test_pairs, gr1, gr2, att1, att2):
    bias_spec = getTermsFromGUI(gr1, gr2, att1, att2)

    g1, g2, a1, a2 = bt_mgr.get_words(bias_spec)
    b_name = rq_mgr.getBiasName(g1, g2, a1, a2)
    print(f"Exporting test pairs for {b_name}")

    fname = f"test_pairs_{b_name}.csv"

    test_pairs.to_csv(fname)
    return gr.File.update(value=fname, visible=True)

# Enable Generation of new sentences, even though not required.
def useOnlineGen(value):
    online_gen_row_update = gr.Row.update(visible=False)
    num_sentences2gen_update = gr.Slider.update(visible=False)
    gen_btn_update = gr.Button.update(visible=False)

    gen_title_update = gr.Markdown.update(visible=False)
    openai_key_update = gr.Textbox.update(visible=False)

    if value == True:
        print("Check is true...")
        online_gen_row_update = gr.Row.update(visible=True)
        num_sentences2gen_update = gr.Slider.update(visible=True)
        gen_btn_update = gr.Button.update(visible=True, value="Generate Additional Sentences")

        gen_title_update = gr.Markdown.update(visible=True)
        openai_key_update = gr.Textbox.update(visible=True)
    else:
        print("Check is false...")

    return (online_gen_row_update,
            num_sentences2gen_update,
            gen_btn_update
            #gen_title_update,
            #openai_key_update,
          )

def changeTerm(evt: gr.EventData):
    global G_CORE_BIAS_NAME

    print("Bias is custom now...")

    G_CORE_BIAS_NAME = None

    return gr.update(interactive=False, visible=False)

def saveBiasTestResult(test_sentences_df, group1, group2, att1, att2, model_name):
  print(f"Saving bias test result...")

  #print(f"Group_1: {group1}")
  #print(f"Group_2: {group2}")
  
  #print(f"Attribute_1: {att1}")
  #print(f"Attribute_2: {att2}")

  print(f"Tested model: {model_name}")
  terms = getTermsFromGUI(group1, group2, att1, att2)
  group1, group2 = bmgr.getSocialGroupTerms(terms)
  att1, att2 = bmgr.getAttributeTerms(terms)

  bias_name = rq_mgr.getBiasName(group1, group2, att1, att2)

  print(f"bias_name: {bias_name}")
  print(f"Terms: {terms}")

  bias_spec_json = {
     "name": bias_name,
     "source": "bias-test-gpt-tool",
     "social_groups": terms['social_groups'],
     "attributes": terms['attributes'],
     "tested_results": {
        "tested_model": model_name
     },
     "templates": [],
     "sentences": []
  }

  bmgr.save_custom_bias(f"{bias_name}.json", bias_spec_json)  

  #return gr.update(value="Bias test result saved!", visible=True)

theme = gr.themes.Soft().set(
    button_small_radius='*radius_xxs',
    background_fill_primary='*neutral_50',
    border_color_primary='*primary_50'
)

soft = gr.themes.Soft(
    primary_hue="slate",
    spacing_size="sm",
    radius_size="md"
).set(
    # body_background_fill="white",
    button_primary_background_fill='*primary_400'
)

css_adds = "#group_row {background: white; border-color: white;} \
               #attribute_row {background: white; border-color: white;} \
               #tested_model_row {background: white; border-color: white;} \
               #button_row {background: white; border-color: white} \
               #examples_elem .label {display: none}\
               #att1_words {border-color: white;} \
               #att2_words {border-color: white;} \
               #group1_words {border-color: white;} \
               #group2_words {border-color: white;} \
               #att1_words_fixed {border-color: white;} \
               #att2_words_fixed {border-color: white;} \
               #group1_words_fixed {border-color: white;} \
               #group2_words_fixed {border-color: white;} \
               #att1_words_fixed input {box-shadow:None; border-width:0} \
               #att1_words_fixed .scroll-hide {box-shadow:None; border-width:0} \
               #att2_words_fixed input {box-shadow:None; border-width:0} \
               #att2_words_fixed .scroll-hide {box-shadow:None; border-width:0} \
               #group1_words_fixed input {box-shadow:None; border-width:0} \
               #group1_words_fixed .scroll-hide {box-shadow:None; border-width:0} \
               #group2_words_fixed input {box-shadow:None; border-width:0} \
               #group2_words_fixed .scroll-hide {box-shadow:None; border-width:0} \
               #tested_model_drop {border-color: white;} \
               #gen_model_check {border-color: white;} \
               #gen_model_check .wrap {border-color: white;} \
               #gen_model_check .form {border-color: white;} \
               #open_ai_key_box {border-color: white;} \
               #gen_col {border-color: white;} \
               #gen_col .form {border-color: white;} \
               #res_label {background-color: #F8FAFC;} \
               #per_attrib_label_elem {background-color: #F8FAFC;} \
               #accordion {border-color: #E5E7EB} \
               #err_msg_elem p {color: #FF0000; cursor: pointer} \
               #res_label .bar {background-color: #35d4ac; } \
               #bloomberg_legend {background: white; border-color: white} \
               #bloomberg_att1 {background: white; border-color: white} \
               #bloomberg_att2 {background: white; border-color: white} \
               .tooltiptext_left {visibility: hidden;max-width:50ch;min-width:25ch;top: 100%;left: 0%;background-color: #222;text-align: center;border-radius: 6px;padding: 5px 0;position: absolute;z-index: 1;} \
               .tooltiptext_right {visibility: hidden;max-width:50ch;min-width:25ch;top: 100%;right: 0%;background-color: #222;text-align: center;border-radius: 6px;padding: 5px 0;position: absolute;z-index: 1;} \
               #filled:hover .tooltiptext_left {visibility: visible;} \
               #empty:hover .tooltiptext_left {visibility: visible;} \
               #filled:hover .tooltiptext_right {visibility: visible;} \
               #empty:hover .tooltiptext_right {visibility: visible;}"

#'bethecloud/storj_theme'
with gr.Blocks(theme=soft, title="Social Bias Testing in Language Models",
               css=css_adds) as iface:
    with gr.Row():
        with gr.Group():
            s1_btn = gr.Button(value="Step 1: Bias Specification", variant="primary", visible=True, interactive=True, size='sm')#.style(size='sm')
            s2_btn = gr.Button(value="Step 2: Test Sentences", variant="secondary", visible=True, interactive=False, size='sm')#.style(size='sm')
            s3_btn = gr.Button(value="Step 3: Bias Testing", variant="secondary", visible=True, interactive=False, size='sm')#.style(size='sm')
    err_message = gr.Markdown("", visible=False, elem_id="err_msg_elem")
    bar_progress = gr.Markdown("     ")

    # Page 1
    with gr.Column(visible=True) as tab1:
        with gr.Column():
            gr.Markdown("### Social Bias Specification")
            gr.Markdown("Use one of the predefined specifications or enter own terms for social groups and attributes")
            with gr.Row():
                example_biases = gr.Dropdown(
                    value="Select a predefined bias to test",
                    allow_custom_value=False,
                    interactive=True,
                    choices=[
                    #"Flowers/Insects <> Pleasant/Unpleasant",
                    #"Instruments/Weapons <> Pleasant/Unpleasant",
                    "Male/Female <> Professions",
                    "Male/Female <> Science/Art",
                    "Male/Female <> Career/Family", 
                    "Male/Female <> Math/Art", 
                    "Eur.-American/Afr.-American <> Pleasant/Unpleasant #1",
                    "Eur.-American/Afr.-American <> Pleasant/Unpleasant #2",
                    "Eur.-American/Afr.-American <> Pleasant/Unpleasant #3",
                    "African-Female/European-Male <> Intersectional",
                    "African-Female/European-Male <> Emergent",
                    "Mexican-Female/European-Male <> Intersectional",
                    "Mexican-Female/European-Male <> Emergent",
                    "Young/Old Name <> Pleasant/Unpleasant",
                    #"Mental/Physical Disease <> Temporary/Permanent",
                    # Custom Biases
                    "Male/Female <> Care/Expertise",
                    "Hispanic/Caucasian <> Treatment-Adherence",
                    "Afr.-American/Eur.American <> Risky-Health-Behaviors"
                    ], label="Example Biases", #info="Select a predefied bias specification to fill-out the terms below."
                )
            with gr.Row(elem_id="group_row"):
                group1 = gr.Textbox(label="Social Group 1", max_lines=1, elem_id="group1_words", elem_classes="input_words", placeholder="brother, father")
                group2 = gr.Textbox(label='Social Group 2', max_lines=1, elem_id="group2_words", elem_classes="input_words", placeholder="sister, mother")
            with gr.Row(elem_id="attribute_row"):
                att1 = gr.Textbox(label='Stereotype for Group 1', max_lines=1, elem_id="att1_words", elem_classes="input_words", placeholder="science, technology")
                att2 = gr.Textbox(label='Anti-stereotype for Group 1', max_lines=1, elem_id="att2_words", elem_classes="input_words", placeholder="poetry, art")
            with gr.Row():
                gr.Markdown("    ")
                get_sent_btn = gr.Button(value="Get Sentences", variant="primary", visible=True)
                gr.Markdown("    ")
    
    # Page 2
    with gr.Column(visible=False) as tab2:
        info_sentences_found = gr.Markdown(value="", visible=False)

        gr.Markdown("### Tested Social Bias Specification", visible=True)
        with gr.Row():
            group1_fixed = gr.Textbox(label="Social Group 1", max_lines=1, elem_id="group1_words_fixed", elem_classes="input_words", interactive=False, visible=True)
            group2_fixed = gr.Textbox(label='Social Group 2', max_lines=1, elem_id="group2_words_fixed", elem_classes="input_words", interactive=False, visible=True)
        with gr.Row():
            att1_fixed = gr.Textbox(label='Stereotype for Group 1', max_lines=1, elem_id="att1_words_fixed", elem_classes="input_words", interactive=False, visible=True)
            att2_fixed = gr.Textbox(label='Anti-stereotype for Group 1', max_lines=1, elem_id="att2_words_fixed", elem_classes="input_words", interactive=False, visible=True)

        with gr.Row():
            with gr.Column():
                additional_gen_check = gr.Checkbox(label="Generate Additional Sentences with ChatGPT (requires Open AI Key)", 
                                            visible=False, interactive=True,
                                            value=False, 
                                            elem_id="gen_model_check")
                with gr.Row(visible=False) as online_gen_row:
                    with gr.Column():
                        gen_title = gr.Markdown("### Generate Additional Sentences", visible=True)

                        # OpenAI Key for generator
                        openai_key = gr.Textbox(lines=1, label="OpenAI API Key", value=None,
                                                placeholder="starts with sk-", 
                                info="Please provide the key for an Open AI account to generate new test sentences",
                                visible=True,
                                interactive=True,
                                elem_id="open_ai_key_box")
                        num_sentences2gen = gr.Slider(1, 20, value=5, step=1, 
                                                interactive=True,
                                                visible=True,
                                                info="Five or more per attribute are recommended for a good bias estimate.",
                                                label="Number of test sentences to generate per attribute", container=True)#.style(container=True) #, info="Number of Sentences to Generate")
                    
                with gr.Row(visible=False) as tested_model_row:
                    with gr.Column():
                        gen_title = gr.Markdown("### Select Tested Model", visible=True)

                        # Tested Model Selection - "openlm-research/open_llama_7b", "tiiuae/falcon-7b"
                        tested_model_name = gr.Dropdown( ["bert-base-uncased","bert-large-uncased","gpt2","gpt2-medium","gpt2-large","emilyalsentzer/Bio_ClinicalBERT","microsoft/biogpt","openlm-research/open_llama_3b","openlm-research/open_llama_7b"], value="bert-base-uncased", 
                            multiselect=None,
                            interactive=True, 
                            label="Tested Language Model", 
                            elem_id="tested_model_drop",
                            visible=True
                            #info="Select the language model to test for social bias."
                        )
            
        with gr.Row():
            gr.Markdown("    ")
            gen_btn = gr.Button(value="Generate New Sentences", variant="primary", visible=True)
            bias_btn = gr.Button(value="Test Model for Social Bias", variant="primary", visible=False)
            gr.Markdown("    ")
        
        with gr.Row(visible=False) as row_sentences:
            with gr.Accordion(label="Test Sentences", open=False, visible=False) as acc_test_sentences:
                test_sentences = gr.DataFrame(
                            headers=["Sentence", "Alternative Sentence", "Group term 1", "Group term 2", "Attribute term"],
                            datatype=["str", "str", "str", "str", "str"],
                            row_count=(1, 'dynamic'),
                            col_count=(5, 'fixed'),
                            interactive=True,
                            visible=True,
                            #label="Generated Test Sentences",
                            max_rows=2,
                            overflow_row_behaviour="paginate")
            
    # Page 3
    with gr.Column(visible=False) as tab3:
        gr.Markdown("### Tested Social Bias Specification")
        with gr.Row():
            group1_fixed2 = gr.Textbox(label="Social Group 1", max_lines=1, elem_id="group1_words_fixed", elem_classes="input_words", interactive=False)
            group2_fixed2 = gr.Textbox(label='Social Group 2', max_lines=1, elem_id="group2_words_fixed", elem_classes="input_words", interactive=False)
        with gr.Row():
            att1_fixed2 = gr.Textbox(label='Stereotype for Group 1', max_lines=1, elem_id="att1_words_fixed", elem_classes="input_words", interactive=False)
            att2_fixed2 = gr.Textbox(label='Anti-stereotype for Group 1', max_lines=1, elem_id="att2_words_fixed", elem_classes="input_words", interactive=False)

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Bias Test Results")
            #with gr.Column(scale=1):
            #    gr.Markdown("### Interpretation")
        with gr.Row():
            with gr.Column(scale=2):
                lbl_model_bias = gr.Markdown("**Model Bias** - % stereotyped choices (↑ more bias)")
                model_bias_label = gr.Label(num_top_classes=1, label="% stereotyped choices (↑ more bias)",
                                            elem_id="res_label",
                                            show_label=False)
                with gr.Accordion("Additional Interpretation", open=False, visible=True):
                    interpretation_msg = gr.HTML(value="Interpretation: Stereotype Score metric details in <a href='https://arxiv.org/abs/2004.09456'>Nadeem'20<a>", visible=False)

                lbl_attrib_bias = gr.Markdown("**Bias in the Context of Attributes** - % stereotyped choices (↑ more bias)")
                #gr.Markdown("**Legend**")
                #attribute_bias_labels = gr.Label(num_top_classes=8, label="Per attribute: % stereotyped choices (↑ more bias)",
                #                                elem_id="per_attrib_label_elem",
                #                                show_label=False)
            #with gr.Column(scale=1):
                with gr.Row():
                    with gr.Column(variant="compact", elem_id="bloomberg_legend"): 
                        gr.HTML("<div style='height:20px;width:20px;background-color:#065b41;display:inline-block;vertical-align:top'></div><div style='display:inline-block;vertical-align:top'> &nbsp; Group 1 more probable in the sentence </div>&nbsp;&nbsp;<div style='height:20px;width:20px;background-color:#35d4ac;display:inline-block;vertical-align:top'></div><div style='display:inline-block;vertical-align:top'> &nbsp; Group 2 more probable in the sentence </div>") 

                with gr.Row():
                    with gr.Column(variant="compact", elem_id="bloomberg_att1"): 
                        gr.Markdown("#### Attribute Group 1")
                        attribute_bias_html_stereo = gr.HTML()
                    with gr.Column(variant="compact", elem_id="bloomberg_att2"):
                        gr.Markdown("#### Attribute Group 2")
                        attribute_bias_html_antistereo = gr.HTML()
            
                gr.HTML(value="Visualization inspired by <a href='https://www.bloomberg.com/graphics/2023-generative-ai-bias/' target='_blank'>Bloomberg article on bias in text-to-image models</a>.")
                save_msg = gr.HTML(value="<span style=\"color:black\">Bias test result saved! </span>", 
                                visible=False)
                
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Accordion("Per Sentence Bias Results", open=False, visible=True):
                    test_pairs = gr.DataFrame(
                            headers=["group_term", "template", "att_term_1", "att_term_2","label_1","label_2"],
                            datatype=["str", "str", "str", "str", "str", "str"],
                            row_count=(1, 'dynamic'),
                            #label="Bias Test Results Per Test Sentence Template",
                            max_rows=2,
                            overflow_row_behaviour="paginate"
                            )
                with gr.Row():
                    # export button 
                    gr.Markdown("    ")
                    with gr.Column():
                        exp_button = gr.Button("Export Test Sentences as CSV", variant="primary")
                        csv = gr.File(interactive=False, visible=False)
                        new_bias_button = gr.Button("Try New Bias Test", variant="primary")
                    gr.Markdown("    ")
        

    # initial interface load 
    #iface.load(fn=loadInterface, 
    #           inputs=[], 
    #           outputs=[openai_key])

    # select from predefined bias specifications
    example_biases.select(fn=prefillBiasSpec, 
                        inputs=None, 
                        outputs=[group1, group2, att1, att2, csv])
    
    # Get sentences
    get_sent_btn.click(fn=retrieveSentences, 
                  inputs=[group1, group2, att1, att2], 
                  outputs=[err_message, online_gen_row, additional_gen_check, num_sentences2gen, 
                           tested_model_row, #tested_model_name, 
                           info_sentences_found, bar_progress, 
                           s1_btn, s2_btn, s3_btn, tab1, tab2, acc_test_sentences, 
                           row_sentences, test_sentences, gen_btn, bias_btn,
                           group1_fixed, group2_fixed, att1_fixed, att2_fixed ])

    # request getting sentences
    gen_btn.click(fn=generateSentences, 
                  inputs=[group1, group2, att1, att2, openai_key, num_sentences2gen], 
                  outputs=[err_message, info_sentences_found, online_gen_row, #num_sentences2gen, 
                           tested_model_row, #tested_model_name, 
                           acc_test_sentences, row_sentences, test_sentences, gen_btn, bias_btn ])
    
    # Test bias
    bias_btn.click(fn=startBiasTest,
                   inputs=[test_sentences,group1,group2,att1,att2,tested_model_name],
                   outputs=[err_message, bar_progress, s1_btn, s2_btn, s3_btn, tab1, tab2, tab3, model_bias_label, 
                            attribute_bias_html_stereo, attribute_bias_html_antistereo, test_pairs, 
                            interpretation_msg, group1_fixed2, group2_fixed2, att1_fixed2, att2_fixed2]
                   )
    
    # top breadcrumbs
    s1_btn.click(fn=moveStep1,
                 inputs=[],
                 outputs=[s1_btn, s2_btn, s3_btn, tab1, tab2, tab3])
    
    # top breadcrumbs
    s2_btn.click(fn=moveStep2,
                 inputs=[],
                 outputs=[s1_btn, s2_btn, s3_btn, tab1, tab2, tab3, additional_gen_check])
    
    # top breadcrumbs
    s3_btn.click(fn=moveStep3,
                 inputs=[],
                 outputs=[s1_btn, s2_btn, s3_btn, tab1, tab2, tab3])
    
    # start testing new bias
    new_bias_button.click(fn=moveStep1_clear,
                          inputs=[],
                          outputs=[s1_btn, s2_btn, s3_btn, tab1, tab2, tab3, group1, group2, att1, att2])


    # Additional Interactions
    #attribute_bias_labels.select(fn=selectAttributeLabel,
    #                             inputs=[],
    #                             outputs=[])
    
    # Editing a sentence
    test_sentences.change(fn=editSentence,
                         inputs=[test_sentences],
                         outputs=[]
                         )

    # tick checkbox to use online generation
    additional_gen_check.change(fn=useOnlineGen, 
                         inputs=[additional_gen_check],
                         outputs=[online_gen_row, num_sentences2gen, gen_btn])#, gen_title, openai_key])

    exp_button.click(export_csv, 
                     inputs=[test_pairs, group1, group2, att1, att2], 
                     outputs=[csv])

    # Changing any of the bias specification terms
    group1.change(fn=changeTerm, inputs=[], outputs=[csv])
    group2.change(fn=changeTerm, inputs=[], outputs=[csv])
    att1.change(fn=changeTerm, inputs=[], outputs=[csv])
    att2.change(fn=changeTerm, inputs=[], outputs=[csv])

iface.queue(concurrency_count=2).launch()