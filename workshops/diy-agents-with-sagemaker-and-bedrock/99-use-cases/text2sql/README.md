---
title: 'Text-to-SQL Agent'
weight: 6
---

# Database RAG & Text to SQL using LangChain & SQL agent

In this lab, you'll explore how to extend RAG capabilities beyond unstructured document retrieval to include structured data sources. You'll learn how to implement text-to-SQL conversion, enabling natural language querying of database information alongside traditional document retrieval. This powerful combination allows for comprehensive enterprise search applications that can access both unstructured documents and structured database records.

## Learning Objectives

By the end of this lab, you will be able to:

- Set up an Athena database and AWS Glue crawler for structured data access
- Implement text-to-SQL capabilities to query databases with natural language
- Create a unified querying experience across documents and databases
- Integrate structured data results with foundation model responses
- Build a comprehensive enterprise search application

## Key Concepts

### Text-to-SQL

Text-to-SQL converts natural language questions into structured SQL queries that can retrieve information from relational databases. This capability bridges the gap between how humans naturally ask questions and how databases store and access data.

### Structured Data Integration

Combining structured database queries with unstructured document retrieval creates a comprehensive knowledge system that can leverage all available data sources in your organization.

### Enterprise Search

Enterprise search applications need to access multiple data sources with different formats. RAG techniques can unify these diverse sources under a single natural language interface.

## Lab Structure

This lab consists of 2 notebooks that guide you through implementing text-to-SQL capabilities:

## text2sql 

1-create-db-tables.ipynb: Configure Athena database and AWS Glue crawler  
2-text2sql-langchain: Use LangChain and SQL agent for text-to-SQL conversion  

## Dataset

In this lab, you'll work with:
- A structured retail transaction dataset stored in Amazon S3
- The dataset will be cataloged using AWS Glue and made queryable through Amazon Athena
- You'll answer business questions that require accessing this structured data

## Getting Started

Before beginning this lab, you should have:
- Basic understanding of SQL and relational databases
- Familiarity with AWS data analytics services

To start working with the notebooks:

1. Navigate to the `workshops/diy-agents-with-sagemaker-and-bedrock/99-use-cases/text2sql` folder in the cloned repository
2. Open `1-create-db-tables` to begin setting up the Athena database and Glue Data Catalog
3. Follow each notebook sequentially to implement text-to-SQL capabilities


## Next Steps

After completing all three labs, you'll have a comprehensive understanding of advanced RAG techniques that combine unstructured document retrieval, metadata filtering, safety guardrails, reranking, and structured data access. These capabilities form the foundation for building sophisticated enterprise AI applications.

Happy learning!