# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:29:30 2025

@author: chdem
"""

import PDAG
import csv

g1 = PDAG.PDAG(v_names=[0, 1, 2], d_edges = [(0, 1), (2, 1)])

g2 = PDAG.PDAG(v_names=[0, 1, 2], u_edges = [ (2, 1)])



M = g1.get_graphical_metrics(g2)




# Example program that generates arrays of 6 elements in a loop
def generate_data():
    for i in range(1, 11):  # Example loop for 10 iterations
        yield [i, i + 1, i + 2, i + 3, i + 4, i + 5]  # Replace with your actual data generation logic

# Open a CSV file for writing
with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Optional: Write a header row
    writer.writerow(['TP', 'FP', 'TN', 'FN', 'TD', 'FD', 'P', 'R', 'F1', 'SHD'])
    
    # Write data rows
    for data_row in generate_data():
        writer.writerow(data_row)
        writer.writerow((1,2,3))
        
    

print("Data written to 'output.csv'")

