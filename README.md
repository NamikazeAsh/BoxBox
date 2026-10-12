# BoxBox: Pit Wall Analytics

Tyre degradation modeling, race strategy simulation, and outcome prediction under the 2026 Formula 1 regulation change.

## Overview

Pit stop strategy is one of the few race variables a Formula 1 team controls in real time. Choosing when to stop and which tyre compound to fit can gain or lose several positions. Teams make these calls with proprietary models, and there is no comparable public tool.

BoxBox is a prerace decision tool with two connected components:

* **Race outcome predictor:** estimates where each driver is likely to finish.
* **Strategy simulator:** shows how different pit strategies (one stop vs. two stop, undercut or overcut timing, responses to safety cars) shift those outcomes for a given driver, track, and starting position.

The 2026 season brought new power unit, aerodynamic, and tyre regulations. This gives us a natural test of distribution shift: do models trained on 2018 to 2025 data still hold for the new cars?

## Data

* [FastF1](https://github.com/theOehrly/Fast-F1): Python interface to the official F1 live timing archive (2018 onward). Lap and sector times, tyre compound and age, pit timings, track status, and weather.
* [Jolpica F1 API](https://github.com/jolpica/jolpica-f1): race results, grids, and pit stops.

All race, qualifying, and practice sessions from 2018 to 2026 are extracted with local caching. Cleaned tables (laps, stints, results, circuits, weather) are stored as Parquet.

Derived quantities:

* **Pit loss per circuit:** time difference between pit laps and normal laps.
* **Long run pace:** from practice stints of 8 or more consecutive laps.
* **Clean lap filter:** removes in and out laps, safety car laps, and laps in traffic.
* **Compounds:** modeled by Pirelli designation (C1, C2, ...) rather than event relative Soft, Medium, and Hard labels.

## Approach

1. **Fuel correction:** remove the effect of fuel burn from each lap time, since cars get faster as fuel burns off.
2. **Tyre degradation:** compare a linear regression of lap time on tyre age against a LightGBM model that also accounts for circuit, track temperature, driver, and team.
3. **Strategy simulation:** use the degradation estimates to compute total race time for each pit strategy, with Monte Carlo simulation for random events such as safety cars.
4. **Outcome prediction:** LightGBM model predicting finishing position from grid position, qualifying pace, and practice pace.
5. **Dashboard:** interactive Streamlit app combining the predictor and simulator.

## Evaluation

* **Degradation models:** lap time prediction error on held out stints.
* **Simulator:** replay historical races with the strategies teams actually used and compare simulated vs. observed gaps and finishing orders, then compare recommendations against actual team decisions.
* **Outcome predictor:** position MAE against a grid position only baseline.
* **Distribution shift:** train on 2018 to 2025, test on completed 2026 races, and test whether the degradation curves differ significantly between eras.

## References

1. FastF1. https://github.com/theOehrly/Fast-F1
2. Jolpica F1 API. https://github.com/jolpica/jolpica-f1
3. FIA. 2026 Formula 1 Technical Regulations, Section C.
4. A. Heilmeier, M. Graf, M. Lienkamp. A race simulation for strategy decisions in circuit motorsports. IEEE ITSC, 2018.
5. A. Heilmeier. Simulation of Circuit Races for the Objective Evaluation of Race Strategy Decisions. Dissertation, Technical University of Munich, 2022.
