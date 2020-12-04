function returnvalue=startup()
addpath '/home/angelo/MATLAB Add-Ons/fieldtrip-20200607';
addpath '/home/angelo/Daten/Master_Arbeit';
addpath(genpath('/home/angelo/Daten/Master_Arbeit/Master_Arbeit'));
mkdir '/home/angelo/Daten/Master_Arbeit/Master_Arbeit/Data/simulation';
ft_defaults
clearvars;
returnvalue=true;