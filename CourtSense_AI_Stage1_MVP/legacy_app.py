"""CourtSense AI Stage I demonstrator. Run: streamlit run app.py"""
import itertools
import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='CourtSense AI', layout='wide')
st.title('🏀 CourtSense AI')
st.caption('Stage I proof of concept • coach-in-the-loop analytics • illustrative data')

with st.sidebar:
    st.header('Input')
    video = st.file_uploader('Upload practice / game video', type=['mp4','mov','avi'])
    st.info('Prototype does not automatically identify basketball events or diagnose injuries.')

st.header('1 · Video intake')
if video:
    st.video(video)
    st.write(f'File: {video.name} | Size: {video.size/1e6:.1f} MB')
else:
    st.write('Upload a video to preview the session. Player statistics below are editable example inputs.')

st.header('2 · Player profiles (editable demo data)')
base = pd.DataFrame([
    ['Ari','PG',.37,.76,.64,.40,28,4],
    ['Ben','SG',.42,.66,.55,.61,35,6],
    ['Chris','SF',.34,.57,.79,.72,31,5],
    ['Drew','PF',.28,.41,.83,.84,39,7],
    ['Eli','C',.12,.25,.90,.91,42,8],
    ['Finn','PG',.31,.85,.58,.45,24,3],
    ['Gabe','SF',.39,.59,.73,.69,34,6],
    ['Hugo','PF',.33,.43,.76,.77,29,5],
], columns=['Player','Role','3PT rate','Ball handling','Defense','Rebounding','Minutes','RPE (1-10)'])
players = st.data_editor(base, num_rows='dynamic', use_container_width=True)

st.header('3 · Explainable five-player lineup suggestions')
st.caption('Transparent illustrative heuristic; not a learned or clinically validated model.')
required=['Player','Role','3PT rate','Ball handling','Defense','Rebounding','Minutes','RPE (1-10)']
try:
    clean=players.dropna(subset=required).copy()
    for col in required[2:]: clean[col]=pd.to_numeric(clean[col])
    if len(clean)<5: st.warning('Add at least five players.')
    else:
        combos=[]
        for idx in itertools.combinations(clean.index,5):
            group=clean.loc[list(idx)]
            spacing=float(group['3PT rate'].mean())
            handling=float(group['Ball handling'].nlargest(2).mean())
            defense=float(group['Defense'].mean())
            rebound=float(group['Rebounding'].mean())
            fatigue=float((group['RPE (1-10)']*group['Minutes']/48).mean()/10)
            score=.24*spacing+.20*handling+.27*defense+.21*rebound-.08*fatigue
            combos.append({'Lineup':', '.join(group['Player'].astype(str)), 'Fit score':round(score,3), 'Spacing':round(spacing,2),'Ball handling':round(handling,2),'Defense':round(defense,2),'Rebounding':round(rebound,2),'Load penalty':round(fatigue,2)})
        rank=pd.DataFrame(combos).sort_values('Fit score',ascending=False).head(10)
        st.dataframe(rank,use_container_width=True,hide_index=True)
        st.write('**Method:** 0.24 × spacing + 0.20 × top-two ball handling + 0.27 × defense + 0.21 × rebounding − 0.08 × load proxy.')
        st.write('Weights are adjustable design assumptions, not evidence of optimal basketball strategy.')
except Exception as e: st.error(f'Please check player inputs: {e}')

st.header('4 · Training load discussion flags')
st.caption('Screening prompts only. RPE is athlete-reported; no injury probability is estimated.')
if 'clean' in locals() and len(clean):
    for _,p in clean.iterrows():
        if p['RPE (1-10)']>=8 and p['Minutes']>=30:
            st.warning(f"{p['Player']}: high reported exertion with substantial minutes. Ask about recovery, soreness and readiness; consult qualified staff if concerned.")
    st.dataframe(clean[['Player','Minutes','RPE (1-10)']],use_container_width=True,hide_index=True)

st.header('5 · Coach review')
notes=st.text_area('Coach observations / corrections',placeholder='e.g., player ID unclear at 02:14; revise event tags')
if st.button('Generate review note'):
    st.success('Review note generated below; copy into your pilot log.')
    st.code('COURTSENSE AI — COACH REVIEW\n'+(notes or 'No corrections entered.')+'\nAll model suggestions require coach review.',language='text')

st.caption('Privacy: use only footage with appropriate consent. Uploaded files are handled by your Streamlit deployment; configure retention and access controls before real use.')
