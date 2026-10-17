#!/usr/bin/env python3
"""Print observation-age stats for eval dirs; flag p95 > 60 ms as DEGRADED."""
import json,glob,sys
for d in sys.argv[1:]:
    ages=[]
    for f in glob.glob(d+'/cohort*/policy_episode_*.json'):
        for o in json.load(open(f)).get('observation_timing',[]) or []:
            a=o.get('ages_sec') or {}
            if a: ages.append(max(a.values()))
    ages.sort()
    if not ages: print(d.split('/')[-1], 'no timing'); continue
    p95=ages[int(.95*len(ages))]*1000
    print(f"{d.split('/')[-1][-40:]} p95 {p95:.0f}ms max {ages[-1]*1000:.0f}ms {'DEGRADED' if p95>60 else 'clean'}")
