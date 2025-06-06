# Structure of the folder

This folder contains the code and details about the factory load disaggregation.
To see the full funcioning of this module, open the notebook 'dataset_loading_notebook.ipynb'
Below there's a description of the dataset and the libraries used.

## Factory load disaggregation dataset

The dataset used is 'Industrial Machines Dataset for Electrical Load Disaggregation' (de Mello Martins et al. 2018), available at the following link 'https://ieee-dataport.org/open-access/industrial-machines-dataset-electrical-load-disaggregation' after registration. Below is the summary of the Dataset abstract.

This dataset contains heavy-machinery data from the Brazilian industrial sector. The data was collected in a poultry feed factory located  in the state of Minas Gerais, Brazil. Its process can be summarized to creating pellets of ration for poultry from corn or soybeans and added nutrients. The factory produces at fullscale over the entire year, thus it has well-behaved usage patterns at any time. It operates from Mondays through Fridays (and occasionally on Saturdays, in case production is below the monthly target) on a daily three-turn shift from 10:00 PM to 05:00 PM. From 05:00 PM to 10:00 PM electricity prices are higher, so the factory is closed. There are three meters measuring distribution circuits: MV/LV transformer, LVDB-2 and LVDB-3. This means that, there are eight GreenAnt meters on the factory measuring appliances, which are:  Pelletizer I (PI); Pelletizer II (PII); Double-pole Contactor I (DPCI); Double-pole Contactor II (DPCII); Exhaust Fan I (EFI); Exhaust Fan II (EFII); Milling Machine I (MI); and Milling Machine II (MII). All machines under LVDB-2 and LVDB-3 work at 380 V and 60 Hz. The factory  As the factory is cyclically producing pellets at full power, each appliance can be modeled as a threestate machine (OFF, NO LOAD ON, FULL LOAD ON). The milling machines were the last machines to be measured. They were only measured over the last 12 days, which corresponds to roughly 10% of the time measurement intervals from other machines.

### Preprocessing

The dataset is in NILM metadata format (https://github.com/nilmtk/nilm_metadata) and thus, we used the python library 'nilmtk' to open and parse it. This library also needs another module, 'nilm_metadata' that was not easy to install, therefore we decided to change the format of the dataset from 'nilm' to 'json' and to keep only the values that are relevant for us.

In fact, we reduced the number of entries since power monitoring was done every 10 minutes roughly but we needed it just every hour, and also we discarded the data of the Double-pole Contactors as they are not power consuming machines, effectively keeping the measurements of the 2 Pelletizers, 2 Exaust fans, 2 Milling machines. Additionally, we shifted the working period of the machines by -12 hours, because the factory uses night shifts but we need day shifts in order to exploit solar panels. Notice that some machines work 24h, so even by shifting the working period, some measurement are unaffected.

So, the 'json' dataset contains one "row" for each timestamp (every hour, for every day where the machines are monitored), and in each row are stated current, voltage and power consumed by each machine. If a machine is not working in a specific timestamp, the corresponding measurement is 'null', and handled by the code as a triple of (0,0,0).

