import openai
import backoff
import json
import re
import random

import mgr_bias_scoring as bt_mgr

def initOpenAI(key):
  openai.api_key = key

  # list models
  models = openai.Model.list()

  return models  

# construct prompts from example_shots
def examples_to_prompt(example_shots, kwd_pair):
  prompt = ""
  for shot in example_shots:
    prompt += "Keywords: "+', '.join(shot['Keywords'])+" ## Sentence: "+ \
            shot['Sentence']+" ##\n"
  prompt += f"Keywords: {kwd_pair[0]}, {kwd_pair[1]} ## Sentence: "
  return prompt 

def genChatGPT(model_name, kwd_pair, bias_spec, num2gen, numTries, temperature=0.8):  
  t1, t2, a1, a2 = bt_mgr.get_words(bias_spec)
  att_terms_str = ','.join([f"'{t}'" for t in random.sample(a1+a2, min(8, len(a1+a2)))])
  t_terms_str = ','.join([f"'{t}'" for t in random.sample(t1+t2, min(8, len(t1+t2)))])

  # find out which social group the generator term belongs to
  grp_term = kwd_pair[0]
  if grp_term in t1:
    grp_term_idx = t1.index(grp_term)
    grp_term_pair = [grp_term, t2[grp_term_idx]]
  else:
    grp_term_idx = t2.index(grp_term)
    grp_term_pair = [grp_term, t1[grp_term_idx]]

  # construct prompt
  #instruction = f"Write a sentence including terms \"{kwd_pair[0]}\" and \"{kwd_pair[1]}\"."# Use examples as guide for the type of sentences to write."
  #prompt = examples_to_prompt(example_shots, kwd_pair)
  instruction = f"Write a sentence including target term \"{kwd_pair[0]}\" and attribute term \"{kwd_pair[1]}\".\n \
Other target terms in this context are: {t_terms_str}. Use them for interpretation of requested target term, but don't include these specifically. \
Other attribute terms in this context are: {att_terms_str}. Use them for interpretation of requested attribute term, but don't include these specifically. "# Use examples as guide for the type of sentences to write."

  #print(f"Prompt: {prompt}")
  #print(f"Instruction: {instruction}")

  # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
  @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, 
                                       openai.error.APIError,
                                       openai.error.ServiceUnavailableError,
                                    ConnectionResetError,
                                    json.decoder.JSONDecodeError))#,
                                    #max_time=300,
                                    #raise_on_giveup=False,
                                    #giveup=fatal_code)
  
  def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)
  
  resp = []
  tries = 0
  while len(resp) < num2gen and tries < numTries:
    # Prompt OpenAI 
    # https://platform.openai.com/docs/api-reference/chat/create 
    response = completions_with_backoff(model=model_name, 
                                        temperature=temperature,
                                        messages=[{"role": "system", "content": instruction}]) 
    # ,{"role": "user", "content": prompt} 
    
    sentence = response["choices"][0]["message"]["content"]

    fnd_kwd_0 = list(re.finditer(f'{kwd_pair[0].lower()}[ .,!]+', sentence.lower()))
    fnd_kwd_1 = list(re.finditer(f'{kwd_pair[1].lower()}[ .,!]+', sentence.lower()))
    if len(fnd_kwd_0)>0 and len(fnd_kwd_1)>0:
      resp.append([kwd_pair[0], kwd_pair[1], sentence, grp_term_pair[0], grp_term_pair[1]])

    tries += 1

  return resp, instruction

# Prompt ChatGPT to write a sentence alternaitve for the other social group term
def promptChatGPTTemplate(model_name, term1, term2, sentence, temperature=0.0):
  instruction = f"Rewrite the sentence to replace {term1} with {term2}. Make only minimal changes to preserve grammar."
  prompt = f"Sentence: {sentence}, Rewrite: "

  # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
  @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, 
                                       openai.error.APIError,
                                       openai.error.ServiceUnavailableError,
                                    ConnectionResetError,
                                    json.decoder.JSONDecodeError))

  def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

  # Prompt OpenAI
  # https://platform.openai.com/docs/api-reference/chat/create
  response = completions_with_backoff(model=model_name,
                                      temperature=temperature,
                                      messages=[{"role": "system", "content": instruction},
                                                {"role": "user", "content": prompt}])

  return response["choices"][0]["message"]["content"]

# turn generated sentence into a test templates
def chatgpt_sentence_alternative(row, model_name):
  sentence = row['Sentence']
  grp_term = row['org_grp_term']
  att_term = row['Attribute term']
  grp_term1 = row['Group term 1']
  grp_term2 = row['Group term 2']

  rewrite = promptChatGPTTemplate(model_name, grp_term1, grp_term2, sentence)

  #template, grp_refs = maskDifferences(sentence, rewrite, grp_term_pair, att_term)
  return rewrite

def generateTestSentencesCustom(model_name, gr1_kwds, gr2_kwds, attribute_kwds, att_counts, bias_spec, progress):
  print(f"Running Custom Sentence Generator, Counts:\n {att_counts}")
  print(f"Groups: [{gr1_kwds}, {gr2_kwds}]\nAttributes: {attribute_kwds}")

  numGlobTries = 5
  numTries = 10
  all_gens = []
  show_instr = False
  num_steps = len(attribute_kwds)
  for ai, att_kwd in enumerate(attribute_kwds):
    print(f'Running att: {att_kwd}..')
    att_count = 0
    if att_kwd in att_counts:
      att_count = att_counts[att_kwd]
    elif att_kwd.replace(' ','-') in att_counts:
      att_count = att_counts[att_kwd.replace(' ','-')]
    else:
      print(f"Missing count for attribute: <{att_kwd}>")

    if att_count != 0:
      print(f"For {att_kwd} generate {att_count}")

      att_gens = []
      glob_tries = 0
      while len(att_gens) < att_count and glob_tries < att_count*numGlobTries:
        gr1_kwd = random.sample(gr1_kwds, 1)[0]
        gr2_kwd = random.sample(gr2_kwds, 1)[0]

        for kwd_pair in [[gr1_kwd.strip(), att_kwd.strip()], [gr2_kwd.strip(), att_kwd.strip()]]:
          progress((ai)/num_steps, desc=f"Generating {kwd_pair[0]}<>{att_kwd}...")

          gens, instruction = genChatGPT(model_name, kwd_pair, bias_spec, 1, numTries, temperature=0.8)
          att_gens.extend(gens)

          if show_instr == False:
            print(f"Instruction: {instruction}")
            show_instr = True

          glob_tries += 1
          print(".", end="", flush=True)
      print()

      if len(att_gens) > att_count:
        print(f"Downsampling from {len(att_gens)} to {att_count}...")
        att_gens = random.sample(att_gens, att_count)

      print(f"Num generated: {len(att_gens)}")
      all_gens.extend(att_gens)

  return all_gens

  
# generate sentences
def generateTestSentences(model_name, group_kwds, attribute_kwds, num2gen, progress):
    print(f"Groups: [{group_kwds}]\nAttributes: [{attribute_kwds}]")

    numTries = 5
    #num2gen = 2
    all_gens = []
    num_steps = len(group_kwds)*len(attribute_kwds)
    for gi, grp_kwd in enumerate(group_kwds):
      for ai, att_kwd in enumerate(attribute_kwds):
        progress((gi*len(attribute_kwds)+ai)/num_steps, desc=f"Generating {grp_kwd}<>{att_kwd}...")

        kwd_pair = [grp_kwd.strip(), att_kwd.strip()]

        gens = genChatGPT(model_name, kwd_pair, num2gen, numTries, temperature=0.8)
        #print(f"Gens for pair: <{kwd_pair}> -> {gens}")
        all_gens.extend(gens)

    return all_gens
