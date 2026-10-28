"""Real video inference using Ultralytics YOLO pose (weights downloaded on first use)."""
from __future__ import annotations
import cv2
import numpy as np
from analytics import extract_pose_metrics, spacing_proxy, SKELETON


def process_video(video_path, output_path, model_name='yolo11n-pose.pt', sample_every=12, max_frames=180):
    from ultralytics import YOLO
    model=YOLO(model_name)
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError('Cannot open video; try MP4 H.264.')
    fps=cap.get(cv2.CAP_PROP_FPS) or 24
    w,h=int(cap.get(3)),int(cap.get(4))
    if w<=0 or h<=0: raise RuntimeError('Invalid video dimensions')
    writer=cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'),max(1.0,fps/sample_every),(w,h))
    if not writer.isOpened(): raise RuntimeError('Unable to create annotated MP4')
    rows=[]; idx=0
    try:
        while len(rows)<max_frames:
            ok,frame=cap.read()
            if not ok:break
            if idx%sample_every:
                idx+=1;continue
            result=model.predict(source=frame,verbose=False,conf=0.30,device='cpu')[0]
            pts=result.keypoints.data.cpu().numpy() if result.keypoints is not None else np.empty((0,17,3))
            boxes=result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0,4))
            centers=[((b[0]+b[2])/2,(b[1]+b[3])/2) for b in boxes]
            for person,k in enumerate(pts):
                for a,b in SKELETON:
                    if k[a,2]>.35 and k[b,2]>.35:
                        cv2.line(frame,tuple(k[a,:2].astype(int)),tuple(k[b,:2].astype(int)),(40,220,80),2)
                for x,y,confidence in k:
                    if confidence>.35:cv2.circle(frame,(int(x),int(y)),3,(255,170,30),-1)
                if person==0:
                    m=extract_pose_metrics(k)
            row={'time_s':round(idx/fps,2),'people_count':len(pts),
                 'screen_spread':spacing_proxy(centers,w)}
            # First visible person is not necessarily the same player across frames.
            row.update(extract_pose_metrics(pts[0]) if len(pts) else {k:None for k in ['left_knee_deg','right_knee_deg','left_elbow_deg','right_elbow_deg','left_hip_deg','right_hip_deg']})
            cv2.putText(frame,f'Detected people: {len(pts)} | {idx/fps:.1f}s',(16,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
            writer.write(frame);rows.append(row);idx+=1
    finally:
        cap.release();writer.release()
    return rows
