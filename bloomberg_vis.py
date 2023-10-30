# def bloombergViz(val, numblocks=10, flip=False):
#     percent = round(val * 100)
#     percentStr = f"{percent}"
#     filled = "<div style='height:20px;width:20px;background-color:#065b41;display:inline-block'></div> "
#     unfilled = "<div style='height:20px;width:20px;background-color:#35d4ac;display:inline-block'></div> "
#     numFilled = round((percent/100) * numblocks)
#     numUnFilled = numblocks - numFilled
#     if flip:
#         return numFilled * unfilled + numUnFilled * filled; 
#     return numFilled * filled + numUnFilled * unfilled

# def att_bloombergViz(att, val, numblocks, flip=False):
#     viz = bloombergViz(val, numblocks, flip)
#     attHTML = f"<div style='border-style:solid;border-color:#999;border-radius:12px'>{att}: {round(val*100)}%<br>{viz}</div><br>"
#     return attHTML

def bloombergViz(att, val, numblocks, score_templates_df, onRight=False, flip=False):
    # percent = round(val * 100)
    # percentStr = f"{percent}"
    # filled = "<div style='height:20px;width:20px;background-color:#555;display:inline-block'><span class='tooltiptext' style='color:#FFF'>{}</span></div> "
    # unfilled = "<div style='height:20px;width:20px;background-color:#999;display:inline-block'><span class='tooltiptext' style='color:#FFF'>{}</span></div> "
    # numFilled = round((percent/100) * numblocks)
    # numUnFilled = numblocks - numFilled

    leftColor = "#065b41" #"#555"
    rightColor = "#35d4ac" #"#999"
    if flip:
        leftColor = "#35d4ac" #"#999"
        rightColor = "#065b41" #"#555"
    res = ""
    spanClass = "tooltiptext_left"
    if onRight:
        spanClass = "tooltiptext_right"
    dfy = score_templates_df.loc[(score_templates_df['att_term'] == att) & (score_templates_df['stereotyped_b'] == 'yes')]
    dfn = score_templates_df.loc[(score_templates_df['att_term'] == att) & (score_templates_df['stereotyped_b'] == 'no')] 
    #print("dfy", dfy)
    #print("dfn", dfn)
    for i in range(len(dfy.index)):
        #print("--GROUP IN BLOOMBERG--")
        groups = dfy.iloc[i, dfy.columns.get_loc("groups_rel")].split("/")
        gr_disp = groups[0]+"&#47;"+groups[1]
        grp_refs = list(dfy.iloc[i, dfy.columns.get_loc("grp_refs")])

        template = dfy.iloc[i, dfy.columns.get_loc("template")]
        for grp_pair in grp_refs:
            #print(f"Item: {grp_pair[0]} - {grp_pair[1]}")
            template = template.replace("[R]", grp_pair[0]+"/"+grp_pair[1], 1)
        
        # template based
        disp = template.replace("[T]", f"[{gr_disp}]") #, 1)
        
        # sentence/alt-sentence based
        #sentence = dfy.iloc[i, dfy.columns.get_loc("sentence")]
        #alt_sentence = dfy.iloc[i, dfy.columns.get_loc("alt_sentence")]
        #disp = f'"{sentence}"/"{alt_sentence}"'

        res += f"<div style='height:20px;width:20px;background-color:{leftColor};display:inline-block;position:relative' id='filled'><span class='{spanClass}' style='color:#FFF'>{disp}</span></div> "
    for i in range(len(dfn.index)):
        groups = dfn.iloc[i, dfn.columns.get_loc("groups_rel")].split("/")
        gr_disp = groups[0]+"&#47;"+groups[1]
        grp_refs = list(dfn.iloc[i, dfn.columns.get_loc("grp_refs")])

        template = dfn.iloc[i, dfn.columns.get_loc("template")]
        for grp_pair in grp_refs:
            #print(f"Item: {grp_pair[0]} - {grp_pair[1]}")
            template = template.replace("[R]", grp_pair[0]+"/"+grp_pair[1], 1)

        # template based
        disp = template.replace("[T]", f"[{gr_disp}]")#, 1)
        
        # sentence/alt-sentence based
        #sentence = dfn.iloc[i, dfn.columns.get_loc("sentence")]
        #alt_sentence = dfn.iloc[i, dfn.columns.get_loc("alt_sentence")]
        #disp = f'"{sentence}"/"{alt_sentence}"'

        res += f"<div style='height:20px;width:20px;background-color:{rightColor};display:inline-block;position:relative' id='empty'><span class='{spanClass}' style='color:#FFF'>{disp}</span></div> "
    return res
    # if flip:
    #     return numFilled * unfilled + numUnFilled * filled; 
    # return numFilled * filled + numUnFilled * unfilled

def att_bloombergViz(att, val, numblocks, score_templates_df, onRight=False, flip=False):
    viz = bloombergViz(att, val, numblocks, score_templates_df, onRight, flip)
    attHTML = f"<div style='border-style:solid;border-color:#999;border-radius:12px'>{att}: {round(val*100)}%<br>{viz}</div><br>"
    return attHTML