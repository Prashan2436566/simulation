from server.server import server
from server.server import model_params
"""
Things to add

adaptive
procreate
cluster of resources
shocks
r&d/ innovation
"""
# ideology
SELECTED_IDEOLOGY = "capitalist" 
#SELECTED_IDEOLOGY = "green_capitalist"
#SELECTED_IDEOLOGY = "socialist"
#SELECTED_IDEOLOGY = "green_socialist"
#SELECTED_IDEOLOGY = "adaptive"

model_params["ideology"] = SELECTED_IDEOLOGY

if __name__ == "__main__":
    server.port = 8523
    server.launch()

    print(f"Running simulation for ideology: {SELECTED_IDEOLOGY}")
