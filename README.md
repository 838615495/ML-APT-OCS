ML-APT: Machine Learning-enhanced Atom Probe Tomography Framework
====

Project Overview
-------
ML-APT is a machine learning framework specifically designed to identify and quantify Ordered Interstitial Complexes (OICs) in medium-entropy alloys. Leveraging advanced in-situ air-doped additive manufacturing, this project integrates Atom Probe Tomography (APT) with machine learning algorithms to achieve high-precision identification and quantitative analysis of nanoscale short-range order (SRO) structures and interstitial complexes.

Research Background
-------
Traditional metal additive manufacturing typically requires expensive ultra-high vacuum or high-purity inert gas environments to suppress impurity-induced embrittlement. This study subverts this paradigm, demonstrating that trace amounts of Oxygen (O) and Nitrogen (N) from the air can be transformed into effective in-situ alloying elements that simultaneously enhance both strength and ductility. Through ML-enhanced APT analysis, we identified two types of nanoscale ordered interstitial complexes: Oxygen-rich OIC1 and Nitrogen-rich OIC2. These act as dislocation pinning points, promoting extensive cross-slip and activating Frank-Read sources during plastic deformation, thereby achieving an exceptional strength-ductility synergy.



Core Functionalities
-------
Multi-scale Structural Identification: Identification ranging from atomic-scale SRO structures to nanoscale OICs.    
Physics-informed Feature Extraction: Utilization of the Poisson-KNN statistical analysis method to describe local chemical correlations more accurately.
Hierarchical Classification Strategy: Integration of Random Forest algorithms with Nearest Neighbor (NN) searches for precise OIC identification.
Statistical Validation: Verification of non-randomness in identified structures using the contingency coefficient (μ).
3D Visualization: Generation of 3D spatial distribution maps for Ti-Zr enriched regions and Nb enriched regions.
Quantitative Analysis: Precise calculation of key parameters including OIC number density, size distribution, and chemical composition.

Technical Features
-------
1. Data Generation & Feature Engineering
Synthetic Structural Database: Includes random BCC solid solutions, Ti-Zr enriched regions, and Nb enriched regions.
Physics-informed Features: Chemical coordination preference parameters are used instead of traditional spatial maps to better reflect local chemical correlations.
Poisson-KNN Statistical Method: Combines K-Nearest Neighbor statistics with Poisson distribution models to provide a more reliable description of local chemical associations.
2. Simulated APT Detection
Noise Simulation: Introduction of noise models consistent with actual APT experiments.
Detection Efficiency Correction: Account for the practical detection efficiency of APT detectors.
Voxelization: Conversion of atomic coordinates into voxel formats suitable for machine learning processing.
3. Machine Learning Models
Random Forest Algorithm: Serves as the core classifier for SRO region identification.
Hierarchical Recognition Strategy: Identifies enriched regions first, then integrates interstitial atoms to form OICs.
Hyperparameter Optimization: Model parameters are optimized via grid search and cross-validation.
4. Prediction & Validation
Nearest Neighbor Search: Incorporates O/N atoms into the identified SRO units.
Statistical Verification: Uses contingency coefficients to validate the non-random nature of the results.
Classification Rules: OICs are categorized into OIC1 and OIC2 based on the occupancy ratio of O/N atoms.
5. Performance MetricsClassification Accuracy: Achieved 96.7% on the test set.ROC-AUC Values: 0.98 for Ti-Zr enriched regions and 0.96 for Nb enriched regions.Cross-validation Consistency: Five-fold cross-validation accuracy standard deviation < 0.5%.Operational Efficiency: Processing time for a single sample (~$10^7$ atoms) is approximately 7 minutes (on an Intel i9-12900H).

