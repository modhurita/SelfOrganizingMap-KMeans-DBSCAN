# SelfOrganizingMap-KMeans-DBSCAN

The goal of this project is to explore the use of the **Self-Organizing Map** (SOM) technique applied to geological data. 
In particular, the aim is to apply it to subsurface resistivity data obtained using airborne electromagnetic methods. 
The dataset is from the province of Drenthe in the Netherlands. 
It turns out that SOM is unsuitable for this task and this dataset, 
but **K-means clustering** followed by **DBSCAN** (density-based spatial clustering of applications with noise) allows 
successful partitioning of the data into high- and low-resistivity regions, with each disjoint low-resistivity region being classified as a separate cluster.
