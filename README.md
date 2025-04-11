# AI for Social Good: Solar Panels project

## Overview
1. Chiamiamo API meteo per forecast sul meteo
2. Funzione di python che converte il meteo in produzione di energia per i nostri pannelli settimanalmente (granularità migliore possibile). ALTERNATIVA AI: Bayesian Network.
3. Dalla nostra produzione di pannelli vogliamo:
   * Minimizzare i consumi di energia importata da Enel 
   * Con i vari vincoli sia hard (non possiamo in un certo momento avere più di TOT kWh) che soft (la gente preferisce fare la lavatrice quando è sveglia magari signora mia non proviamo a svegliarci alle 3 di notte solo per far partire una cazzo di lavatrice ma di che stiamo parlando). CSP AI.
    COME? Andando a cambiare l'orario di uso degli elettrodomestici.
4. SERVE LA BATTERIA?
   * Problema senza batteria. Poi possiamo fare il confronto tra quanto consumiamo e quanto produciamo coi pannelli e sappiamo che potremmo usare una batteria.
   * Per quelli che hanno bisogno di batterie: STABLE MATCHING tra persone e batteria?????? (persone vogliono pagare meno e la batteria vuole un profilo di consumo). E rifacciamo il problema di minimo. A questo punto confrontiamo quanta batteria risparmi in tot anni.

SUDDIVISIONE:
CSP: Fra e Fede
Bayesian network: Gian e Matte

# PART 1 (panels production)

## Weather APIs
* [Open Meteo API](https://open-meteo.com/): good starting point, free and open source and without the key, it's the one I showed you this morning.
* [World Weather API](https://www.weatherapi.com/weather/): free trial of 14 days.
* [OpenWeather API](https://openweathermap.org/api): not everything is free, but something is.

## Datasets to train our solar panels production model
* [Github of many datasets](https://github.com/Charlie5DH/Solar-Power-Datasets-and-Resources)
* [Live data of solar panels production in UK](https://www.solar.sheffield.ac.uk/pvlive/)
* [Weather and corresponding solar panels production](https://catalog.data.gov/dataset/nist-campus-photovoltaic-pv-arrays-and-weather-station-data-sets-05b4d)
* [Same as point above but in UK](https://data.london.gov.uk/dataset/photovoltaic--pv--solar-panel-energy-generation-data)

To understand what to use:
* [Models for the task](https://pvpmc.sandia.gov/)
* [Performance indicators](https://trackso.in/knowledge-base/key-performance-indicators-for-solar-pv-plants/)
* [Tutorial on YT](https://www.youtube.com/watch?v=thYLG4JmaFI) with corresponing [notebook](https://towardsdatascience.com/solar-panel-power-generation-analysis-7011cc078900/)
* [Other tutorial](https://www.youtube.com/watch?v=sweUakFg3I8)
* [Last tutorial I swear](https://www.youtube.com/watch?v=gNgKSduzDLY)

Practically:
* [Python library to do everything](https://pvlib-python.readthedocs.io/en/stable/)

# PART 2 (CSP)
CSP MODEL: [Overleaf project](https://www.overleaf.com/8135128266hbngvgtbdngj#1c1cb2)

We can model the constraints and use CSP or even an informed search if we can obtain a good heuristic and test to see which one performs better (in general it should be the heuristic one since we have problem-specific heuristics).
CSP involves discrete variables with finite domain (since every minute can be used).
⇒ tutti i CSP possono essere convertiti in CSP con constraints binari: è un timetabling problem
which class is offered when and where?
HOW TO SOLVE IT:
- Full search
- Backtracking with some improvements on how to choose variables and order, backjumping, no-good, forward checking, constraint propagation (arc consistency)
- Local search: utile se vogliamo fare delle modifiche minime al volo quando cambiano dei requirements
Anche un SOFT CSP: ogni assignment di valori è associato a un preference value. Di fatto invece che dire “Non posso fare questa cosa” dico “se faccio questa cosa a questa ora è meglio”. Si potrebbe vedere come un Weighted CSP o anche un multi-criteria problem (vogliamo massimizzare/minimizzare due cose assieme)… attenzione che viene fuori anche BRANCH AND BOUND.
Vedi anche CP net

### 1. Real world dataset on machine energy consumption in industry
TODO: understand which one is better:
* https://www.nrel.gov/docs/fy24osti/90442.pdf
* https://ieee-dataport.org/documents/hourly-energy-consumption-industrial-site
* https://github.com/creators01/steel-industry-energy-dataset

### 2. Constraints
Objective function:
* Minimize imported energy
* Minimize cost of imported energy (if this cost changes during the day) 

Variables: 
* the starting time of every machine. We can suppose that once started it never stops or that it could be paused
* maybe more than one machine for a certain type: this could make things worse

Ideas:
* Time window constraints: some machines may only be allowed to operate during specific times due to labor laws, noise restrictions, or dependencies.
* Mutual exclusion constraints: Some machines cannot run simultaneously due to power load limits or shared resources.
* Max power load per time slot: To avoid peak demand surcharges or circuit overloads, enforce a maximum allowed total consumption per hour. (maybe also the limit could be variable). SOFT CONSTRAINT
* Machine dependencies: Some machines can only start after others finish.
* Minimum/Maximum runtime constraints: Some machines must run at least or at most a certain number of times per day.
* Worker availability: model operator shifts if some machines require manual control.
* The cost of imported energy depends on the hour
* Setup Time or Cooldown Period: Some machines require a startup or cooldown phase between uses.
* Maintenance Windows / Downtime: Certain hours or days are unavailable due to planned maintenance.
* Shared Energy Infrastructure: Some machines might share a line or substation. Their combined use must not exceed the line’s capacity.
* Cyclic operations: Some tasks need to be repeated daily or periodically, not only 3 times a week but at a certain distance
* Priority machines
* Job deadlines: Some processes must finish before a specific time
* Demand Response Signals: If your industry participates in demand response, it may need to reduce load during certain times.

Soft constraints:
* User Preferences or Soft Time Windows

Per la parte 2:
* Battery storage: If the site has batteries, we can model:
  * Charging/discharging rates
  * Battery state of charge
  * Charging only during surplus solar hours


### 3. Algorithms to solve the problem
See what we did in class