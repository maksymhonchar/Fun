Shelf Space Planning

1. Symphony RetailAI
        - balancing days of supply
        - merchandising objectives are adhered to
        - providing products with necessary capacity levels
        - ensures category intelligence makes its way to the shelf

2. Youtube excel project - Store Shelf Space Optimizer
        - https://www.youtube.com/watch?v=eLLnGU0HsV4
        - INPUT data needed: department; store; start date; end date;
        - OUTPUT: shelves started with - ended with; shelves gained/lost; sales gain/loss
                - -> for each category
3. Youtube vid 2: Space Allocation Optimization
        - https://www.youtube.com/watch?v=OZP3pX_PiG0
        - https://www.youtube.com/watch?v=ql8FiJ0Az4A
        - Motivation: 
                - GOAL: model that captures the relevant criteria
                        - reducing infrastructure costs
                        - improving perform
        - Generalization:
                - Consumers
                - Spaces
                - Resources (area, jacks, bandwidth, power)
        - Constraints
                - Required resource
                - Space compatibility
                - Consumer compatibility
        - Metrics
                - $ move cost
                - office area (too much / too low for each person)
                - synergy (how 2 employees work with each other on distance)
        - Data
                - Multiple, dynamic data sources
                - Snapshot, resolve, reconcile
                - complex rules to map source data
        - Search:
                - Allocation == mapping consumers to space
                        - there are many possible mappings
        - Divide and Conquer
                - Constraint solving - find nearest constrained solution; blind process, extensible
                - greedy heuristic - priority queue driven, local optimum
        - Random Allocation - ?


------

Bin packing algorithms

1. The first-fit algorithm
        - keep the weights in the order they are given and pack them one at a time
        - keep the bins in order also
        - given a weight, check each bin, in order, until you find one that has room for the weight, and put the weight in that bin

2. The best-fit algorithm
        - keep the weights in the order they are given, and pack them one at a time
        - Place each weight into the bin that is the "best fit" -> that has the least capacity remaining (but still enough room to fit the weight)
        
        
https://arxiv.org/pdf/1702.04415.pdf
https://www.researchgate.net/publication/281652656_Optimizing_Inventory_Replenishment_and_Shelf_Space_Management_in_Retail_Stores
https://www.researchgate.net/publication/304285038_A_Retail_Shelf-space_and_In-store_Process_Optimization_Model