---
title: 'Text-to-DSL: Query OpenSearch with natural language using GenAI and MCP'
weight: 6
---

# No-SQL RAG & Text to DSL using LangGraph, OpenSearch Serverless and MCP

In this lab, you'll explore how to extend RAG capabilities to NoSQL databases. Using a LLM powered text-to-DSL conversion and a Model Context Protocol(MCP) server which provides a connctivity to a OpenSearch Serverless Collection, this lab shows natural language querying of JSON documents with complex querying conditions. This powerful combination allows for comprehensive enterprise search applications that can access semi-structured data without transforming the data to structured format or generating embeddings using additional LLM costs.

## Learning Objectives

By the end of this lab, you will be able to:

- Set up a serverless collection in Amazon OpenSearch Service
- Ingest a sample findings from [Amazon GuardDuty](https://docs.aws.amazon.com/guardduty/latest/ug/sample_findings.html) to an OpenSearch index
- Create a simple MCP server which retrives documents from OpenSearch indices
- Using LangChain MCP Adapters, connect your agentic application to MCP servers
- Implement text-to-DSL capabilities to query OpenSearch indicies with natural language
- Integrate the query results with foundation model responses
- Build a comprehensive enterprise search application

## Key Concepts

### Text-to-DSL

Text-to-DSL refers to the process of translating natural language text into a Domain-Specific Language (DSL). This is commonly used in the context of search engines like Elasticsearch or OpenSearch where the DSL is used to construct complex search queries. Essentially, you input text that describes what you want to search for, and the system converts it into a DSL query that the search engine can understand and execute.

### Model Context Protocol (MCP)

The Model Context Protocol (MCP) is an open protocol that enables seamless integration between LLM applications and external data sources and tools. Whether you're building an AI-powered IDE, enhancing a chat interface, or creating custom AI workflows, MCP provides a standardized way to connect LLMs with the context they need.

### MCP Server
An MCP Server is a lightweight program that exposes specific capabilities through the standardized Model Context Protocol. Host applications (such as chatbots, IDEs, and other AI tools) have MCP clients that maintain 1:1 connections with MCP servers. MCP servers can access local data sources and remote services to provide additional context that improves the generated outputs from the models.

### LangChain MCP Adapters
The library provides a lightweight wrapper that makes MCP tools compatible with LangChain and LangGraph.


## Lab Structure
* **text2dsl-mcp.ipynb** <br/>
The main notebook filethat guide you through implementing text-to-DSL capabilities

* **cfn-oss-collection.yaml** <br/>
The CloudFormation template to deploy a Amazon OpenSearch Service Serverless Collection to be used in this lab.<br/>
_You can skip this if you are participating an AWS Instructor-led workshop event because this stack is pre-deployed in the provided AWS account_

* **mcp_dsl_server.py** <br/>
The Python script with MCP server implementation. The MCP server provides tools to get a schema of indices of OpenSearch collection, and to execute DSL queries go retrive data from the indicies.

* **utils.py** <br/>
A collection of helper functions to use in this lab.

* **guardduty-index-schema.json** <br/>
This file defines the schema of test dataset being used in the lab. The lab uses sample findings generated from Amazon GuardDuty. 

## Dataset

In this lab, we will use [Amazon GuardDuty](https://aws.amazon.com/guardduty/) to generate JSON documents. Amazon GuardDuty uses AI and ML with integrated threat intelligence from AWS and leading third parties to help protect your AWS accounts, workloads, and data from threats.<br/>
Amazon GuardDuty helps you generate sample findings to visualize and understand the various finding types that it can generate. When you generate sample findings, GuardDuty populates your current findings list with one sample for each supported finding type, including attack sequence finding types.

## Getting Started

Before beginning this lab, you should have:
- Basic understanding in [Query DSL](https://docs.opensearch.org/docs/latest/query-dsl/)
- Familiarity with AWS OpenSearch Service

To start working with the notebooks:

1. Navigate to the `workshops/diy-agents-with-sagemaker-and-bedrock/99-use-cases/text2dsl-mcp` folder in the cloned repository
2. Open `text2dsl-mcp.ipynb` and follow each notebook sequentially to implement text-to-DSL capabilities using MCP
