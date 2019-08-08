# Spark

- Spark is a big data solution that has been proven to be easier and faster than Hadoop MapReduce
- In the era of big data, practitioners need more than ever fast and reliable tools to process streaming of data. Earlier tools like MapReduce were favorite but were slow. To overcome this issue, Spark offers a solution that is both fast and general-purpose
- The main difference between Spark and MapReduce is that Spark runs computations in memory during the later on the hard disk.
- It allows high-speed access and data processing, reducing times from hours to minutes.

# How does Spark work
- Spark is based on computational engine, meaning it takes care of the scheduling, distributing and monitoring application.
- Each task is done across various worker machines called computing cluster.
- A computing cluster refers to the division of tasks.
- One machine performs one task, while the others contribute to the final output through a different task.
- In the end, all the tasks are aggregated to produce an output.

- A significant feature of Spark is the vast amount of built-in library, including MLlib for machine learning.
- Spark is also designed to work with Hadoop clusters and can read the broad type of files, including Hive data, CSV, JSON, Casandra data among other.

- Regular machine learning projects are built around the following methodology:
    - Load the data to the disk
    - Import the data into the machine's memory
    - Process/analyze the data
    - Build the machine learning model
    - Store the prediction back to disk

- Take users recommendation for instance.
- Recommenders rely on comparing users with other users in evaluating their preferences.
- If the data practitioner takes only a subset of the data, there won't be a cohort of users who are very similar to one another.
- Recommenders need to run on the full dataset or not at all.

- Pyspark gives the data scientist an API that can be used to solve the parallel data proceedin problems
- Pyspark handles the complexities of multiprocessing, such as distributing the data, distributing code and collecting output from the workers on a cluster of machines.

- Spark can run standalone but most often runs on top of a cluster computing framework such as Hadoop
- In test and development, however, a data scientist can efficiently run Spark on their development boxes or laptops without a cluster

- One of the main advantages of Spark is to build an architecture that encompasses data streaming management, seamlessly data queries, machine learning prediction and real-time access to various analysis.
- Spark works closely with SQL language, i.e., structured data. It allows querying the data in real time.

# HADOOP
- Hadoop is a framework that allows you to store and process large data sets in parallel and distributed fashion.
- Core components of Hadoop (2):
    1. HDFS: allows to dump any kind of data across the cluster
    2. YARN: allows parallel processing of the data stored in HDFS

# Spark
- Apache Spark is an open-source cluster-computing framework for **real-time processing**
- Provides an interface for programming entire clusters with implicit data parallelism and fault-tolerance.
- Built on top of YARN and it EXTENDS the YARN model to efficiently use more types of computations.
- Happens in memory - really fast and suitable to realtime processing

# Architecture
- Hadoop: NameNoded -> DataNode1, DataNode2, ...
- Spark: Master -> Worker/Slave1, ...2, ...

# Spark complementing Hadoop
- Spark processes data 100 times faster than MapReduce
    - == faster analytic
- Spark Applications can run on YARN leveraging Hadoop cluster
    - == cost optimization
- Apache Spark can use HDFS as its storage
    - == avoid duplication

- Combining Spark's ability (high processing speed, advance analytics, ML capabilities, multiple integration support) with Hadoop's low cost operation on commodity hardware gives the best results.

# HDFS
- HDFS creates an abstraction layer over the distributed storage resources, from where **we can see the whole HDFS as a single unit**

# NameNode
- Is a master daemon
- Maintains and manages DataNodes
- Records metadata (e.g. location of blocks stored, the size of the files, permissions, hierarchy, etc.)
- Receives heartbeat and block report from all the DataNodes

# Secondary NameNode
- **Checkpointing** 
    - == process of combining edit logs with FsImage
- Allows faster Failover as we have a back up of the metadata
- Checkpointing happens periodically (default: 1 hour)

# DataNode
- Slave daemons
- Stores actual data
- Serves read and write requests

# HDFS Data Block
- Each file is stored on HDFS as block
- The default size of each block is 128MB

- Let us say, I have a file example.txt of size 380MB:
    - Block 1: 128MB
    - Block 2: 128MB
    - Block 3: 124MB (MBs remaining)

- QUESTION: how many blocks will be created if a file of size 500MB is copied to HDFS?
    - 4 blocks: 3x 128MB + 116MB

# HDFS Block replication
- Each data blocks are replicated (thrice by default) and are distributed across different DataNodes

- Process:
    1. 248 MB file
    2. Divide it into blocks: block1 128MB, block2 120 MB
    3. (NameNode <=> Secondary NameNode) -> DataNode1, DataNode2, DataNode3
        - btw: replication factor=3
        - DataNode1: block1, block2
        - DataNode2: block1, block2
        - DataNode3: block1, block2

# Rack awareness algorithm
- RA algo reduces latency as well as provides fault tolerance by replicating data block
- RA algo - first replica of a block will be stored on a local rack & the next 2 replicas will be stored on a different (remote) rack