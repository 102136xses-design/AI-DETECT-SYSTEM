import itertools
import pandas as pd

def rank_lineups(players):
    columns=['Player','3PT rate','Ball handling','Defense','Rebounding','Minutes','RPE (1-10)']
    clean=players.dropna(subset=columns).copy()
    for col in columns[1:]:clean[col]=pd.to_numeric(clean[col])
    if len(clean)<5:raise ValueError('At least five complete player profiles required')
    if len(clean)>18:raise ValueError('Limit MVP to 18 players to avoid combinatorial explosion')
    records=[]
    for ids in itertools.combinations(clean.index,5):
        g=clean.loc[list(ids)]
        spacing=g['3PT rate'].mean();handling=g['Ball handling'].nlargest(2).mean()
        defense=g['Defense'].mean();reb=g['Rebounding'].mean()
        load=(g['RPE (1-10)']*g['Minutes']/48).mean()/10
        score=.24*spacing+.20*handling+.27*defense+.21*reb-.08*load
        records.append({'Lineup':', '.join(g['Player'].astype(str)),'Fit score':round(score,3),'Spacing':round(spacing,2),'Handling':round(handling,2),'Defense':round(defense,2),'Rebounding':round(reb,2),'Load proxy':round(load,2)})
    return pd.DataFrame(records).sort_values('Fit score',ascending=False)
