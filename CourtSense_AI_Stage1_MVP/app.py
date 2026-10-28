"""CourtSense AI Stage I: working pose inference + explainable basketball demo."""
import tempfile
from pathlib import Path
import cv2
import pandas as pd
import streamlit as st
from analytics import summarize_frames, movement_observations
from video_ai import process_video

st.set_page_config(page_title='CourtSense AI | Stage I',layout='wide')
st.title('🏀 CourtSense AI')
st.caption('Stage I functional prototype | Actual computer-vision inference | Human-reviewed coaching support')
st.info('This MVP automatically detects human poses and measures joint angles. It does NOT automatically recognize basketball plays, track named athletes, predict injuries, or diagnose medical conditions.')

with st.sidebar:
    st.header('Inference settings')
    stride=st.slider('Analyze every Nth frame',5,40,12)
    limit=st.slider('Maximum analyzed frames',20,300,100,step=20)
    st.caption('First run downloads public YOLO pose model weights. Internet required for initial download.')

st.header('1 · Upload basketball practice or game footage')
video=st.file_uploader('MP4 / MOV / AVI',type=['mp4','mov','avi'])
if video:
    st.video(video)
    if st.button('Run real AI video analysis',type='primary'):
        with st.spinner('Running pose detection on sampled frames (CPU may take a few minutes)...'):
            try:
                with tempfile.TemporaryDirectory() as td:
                    inp=Path(td)/('input'+Path(video.name).suffix.lower())
                    out=Path(td)/'annotated.mp4'
                    inp.write_bytes(video.getvalue())
                    rows=process_video(str(inp),str(out),sample_every=stride,max_frames=limit)
                    st.session_state['video_rows']=rows
                    st.session_state['annotated']=out.read_bytes()
                st.success(f'Analyzed {len(rows)} video frames with real pose inference.')
            except Exception as e:
                st.error(f'Video analysis failed: {e}')
                st.info('Confirm Python 3.10–3.12, dependencies, network access for the first model download, and a readable MP4.')

rows=st.session_state.get('video_rows',[])
if rows:
    summary=summarize_frames(rows)
    c1,c2,c3=st.columns(3)
    c1.metric('Frames analyzed',summary['frames'])
    c2.metric('Mean people detected',summary['mean_people'])
    c3.metric('Median left knee angle',f"{summary['metrics']['left_knee_deg']}°" if summary['metrics']['left_knee_deg'] is not None else 'N/A')
    st.subheader('Pose detection result')
    st.video(st.session_state['annotated'])
    st.download_button('Download annotated MP4',st.session_state['annotated'],file_name='courtsense_annotated.mp4',mime='video/mp4')
    df=pd.DataFrame(rows)
    st.dataframe(df,use_container_width=True)
    st.download_button('Download frame-by-frame measurements (CSV)',df.to_csv(index=False).encode(),file_name='pose_metrics.csv',mime='text/csv')
    st.subheader('Technique review prompts (not injury assessment)')
    for msg in movement_observations(summary['metrics']):st.write('• '+msg)
    st.warning('2D camera angle, occlusion and identity switching affect these measurements. Joint angles alone cannot determine injury risk or fatigue.')

st.header('2 · Coach-assisted tactical tagging')
st.write('Automatic player identity and tactical-play classification are NOT part of this MVP. Log verified events to produce a transparent opponent tendency summary.')
if 'events' not in st.session_state:st.session_state['events']=pd.DataFrame(columns=['Team','Player','Event','Result'])
with st.form('event_entry'):
    a,b,c,d=st.columns(4)
    team=a.selectbox('Team',['Our team','Opponent'])
    player=b.text_input('Player ID / jersey',value='4')
    event=c.selectbox('Event',['3PT attempt','2PT attempt','Drive','Pick-and-roll','Turnover','Rebound'])
    result=d.selectbox('Outcome',['Success','Miss / unsuccessful'])
    if st.form_submit_button('Add reviewed event'):
        st.session_state['events']=pd.concat([st.session_state['events'],pd.DataFrame([{'Team':team,'Player':player,'Event':event,'Result':result}])],ignore_index=True)
events=st.session_state['events']
if len(events):
    st.dataframe(events,use_container_width=True)
    counts=events.groupby(['Team','Player','Event']).size().reset_index(name='Count').sort_values('Count',ascending=False)
    st.subheader('Observed event tendencies (small-sample descriptive statistics)')
    st.dataframe(counts,use_container_width=True)
    st.download_button('Download tactical tags',events.to_csv(index=False).encode(),file_name='tactical_tags.csv',mime='text/csv')

st.header('3 · Explainable five-player lineup ranking')
from lineup import rank_lineups
sample=pd.DataFrame([
['Ari','PG',.37,.76,.64,.40,28,4],['Ben','SG',.42,.66,.55,.61,35,6],
['Chris','SF',.34,.57,.79,.72,31,5],['Drew','PF',.28,.41,.83,.84,39,7],
['Eli','C',.12,.25,.90,.91,42,8],['Finn','PG',.31,.85,.58,.45,24,3],
['Gabe','SF',.39,.59,.73,.69,34,6]],columns=['Player','Role','3PT rate','Ball handling','Defense','Rebounding','Minutes','RPE (1-10)'])
players=st.data_editor(sample,use_container_width=True,num_rows='dynamic')
try:
    rankings=rank_lineups(players)
    st.dataframe(rankings.head(10),use_container_width=True,hide_index=True)
    st.caption('Illustrative heuristic; NOT trained on game outcomes. The highest score is not a proven optimal lineup.')
except Exception as e:st.warning(f'Enter at least five valid players: {e}')

st.header('4 · Workload discussion, not medical diagnosis')
st.write('RPE and minutes are manually entered. Ask athletes about recovery if exertion and workload are high; this does not estimate injury probability.')
for _,p in players.iterrows():
    try:
        if float(p['RPE (1-10)'])>=8 and float(p['Minutes'])>=30:
            st.warning(f"{p['Player']}: review recovery and perceived exertion with coaching/medical staff.")
    except (ValueError,TypeError):pass
st.caption('Consent required for identifiable footage. Demo files are temporary during processing; hosting and privacy policies depend on deployment.')
