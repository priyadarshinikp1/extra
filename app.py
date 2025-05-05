import os
import tempfile
import requests
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pyvis.network import Network
import networkx as nx
from gseapy import enrichr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap
import streamlit.components.v1 as components

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(page_title="Multi-Omics App", layout="wide")
st.image("https://raw.githubusercontent.com/priyadarshinikp1/Multiomics-Integrator-app/main/logo.png", width=200)
st.title("🧬 Multi-Omics Integration TPM Vizzhy App")

with st.sidebar:
    st.markdown("---")
    st.markdown("**👨‍💻 Created by: PRIYADARSHINI**")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/priyadarshini24) | [GitHub](https://github.com/priyadarshinikp1)")

# -----------------------------
# File Upload Section
# -----------------------------
st.header("📁 Upload Omics Data")

genomics = st.file_uploader("Upload Genomics CSV", type="csv")
transcriptomics = st.file_uploader("Upload Transcriptomics CSV", type="csv")
proteomics = st.file_uploader("Upload Proteomics CSV", type="csv")

if genomics:
    gdf = pd.read_csv(genomics)

if transcriptomics:
    tdf = pd.read_csv(transcriptomics)

if proteomics:
    pdf = pd.read_csv(proteomics)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("⚙️ Settings")

cadd_thresh = float(st.sidebar.text_input("Min CADD Score (Genomics)", value="20"))
tpm_thresh = float(st.sidebar.text_input("Min TPM (Transcriptomics)", value="1"))
p_intensity_thresh = float(st.sidebar.text_input("Min Intensity (Proteomics)", value="1000"))

run_enrichment = st.sidebar.checkbox("Run Enrichment Analyses", value=True)
show_network = st.sidebar.checkbox("Show Network Visualization", value=True)
show_association_table = st.sidebar.checkbox("Show Association Table", value=True)
num_pathways_to_show = st.sidebar.slider("Number of Pathways to Display in Network", min_value=1, max_value=100, value=10)

# -----------------------------
# Preview Filtered Data
# -----------------------------
preview_n = st.sidebar.slider("Preview Top N Filtered Rows", 5, 50, 10)

st.subheader("🔍 Filtered Data Preview")

if genomics and transcriptomics and proteomics:
    try:
        gdf['CADD'] = pd.to_numeric(gdf['CADD'], errors='coerce')
        tdf['TPM'] = pd.to_numeric(tdf['TPM'], errors='coerce')
        pdf['Intensity'] = pd.to_numeric(pdf['Intensity'], errors='coerce')

        gdf_filtered = gdf[gdf['CADD'] >= cadd_thresh]
        tdf_filtered = tdf[tdf['TPM'] >= tpm_thresh]
        pdf_filtered = pdf[pdf['Intensity'] >= p_intensity_thresh]

        st.markdown("**Genomics**")
        st.dataframe(gdf_filtered.head(preview_n))
        st.markdown("**Transcriptomics**")
        st.dataframe(tdf_filtered.head(preview_n))
        st.markdown("**Proteomics**")
        st.dataframe(pdf_filtered.head(preview_n))

    except Exception as e:
        st.error(f"Integration error: {e}")

# -----------------------------
# Enrichment Analysis
# -----------------------------
enrichment_results = {}

if run_enrichment:
    st.header("📊 Enrichment Analyses")
    libraries = {
        "Reactome Pathways": "Reactome_2016",
        "Disease Associations": "OMIM_Disease",
        "HMDB Metabolites": "HMDB_Metabolites"
    }

    common_genes = set(gdf_filtered['Gene']) & set(tdf_filtered['Gene']) & set(pdf_filtered['Gene'])
    
    # Display the common genes
    if common_genes:
      st.subheader("🧬 Common Genes Found")
      st.write(f"Total Common Genes: {len(common_genes)}")
      st.dataframe(pd.DataFrame(list(common_genes), columns=["Gene"]))
    else:
      st.warning("⚠️ No common genes found after filtering. Please adjust your thresholds.")
    
    for name, lib in libraries.items():
        try:
            enr = enrichr(gene_list=list(common_genes), gene_sets=lib, outdir=None)

            if enr.results.empty:
                continue

            enrichment_results[name] = enr.results

            st.subheader(f"{name} Enrichment Results")
            st.dataframe(enr.results)

            fig = px.bar(enr.results.head(10), x="Term", y="Combined Score", title=f"Top 10 {name}")
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error in {name}: {e}")

# -----------------------------
# Network Visualization
# -----------------------------
if show_network and enrichment_results:
    st.subheader("🧠 Interactive Omics Network")
    net = Network(height='800px', width='100%', directed=False, notebook=False)
    net.force_atlas_2based()

    # Create a set to track nodes added
    added_nodes = set()

    # Mapping enrichment sources
    reactome_df = enrichment_results.get("Reactome Pathways", pd.DataFrame())
    hmdb_df = enrichment_results.get("HMDB Metabolites", pd.DataFrame())
    disgenet_df = enrichment_results.get("Disease Associations", pd.DataFrame())

    # Add Gene nodes
    for gene in common_genes:
        if gene not in added_nodes:
            net.add_node(gene, label=gene, color='gray', title=f"Gene: {gene}")
            added_nodes.add(gene)

    # Add Protein nodes
    for _, row in pdf_filtered.iterrows():
        gene = row['Gene']
        protein = row['Protein']
        if gene in common_genes:
            if protein not in added_nodes:
                net.add_node(protein, label=protein, color='gold', title=f"Protein: {protein}")
                added_nodes.add(protein)
            net.add_edge(gene, protein)

    # Add Pathways
    if not reactome_df.empty:
        for _, row in reactome_df.iterrows():
            pathway = row['Term']
            genes_in_pathway = row['Genes'].split(';')
            if pathway not in added_nodes:
                net.add_node(pathway, label=pathway, color='skyblue', title=f"Pathway: {pathway}")
                added_nodes.add(pathway)
            for gene in genes_in_pathway:
                if gene in common_genes:
                    net.add_edge(gene, pathway)

    # Add Metabolites
    if not hmdb_df.empty:
        for _, row in hmdb_df.iterrows():
            metabolite = row['Term']
            genes_in_metabolite = row['Genes'].split(';')
            if metabolite not in added_nodes:
                net.add_node(metabolite, label=metabolite, color='green', title=f"Metabolite: {metabolite}")
                added_nodes.add(metabolite)
            for gene in genes_in_metabolite:
                if gene in common_genes:
                    net.add_edge(gene, metabolite)

    # Add Diseases
    if not disgenet_df.empty:
        for _, row in disgenet_df.iterrows():
            disease = row['Term']
            genes_in_disease = row['Genes'].split(';')
            if disease not in added_nodes:
                net.add_node(disease, label=disease, color='red', title=f"Disease: {disease}")
                added_nodes.add(disease)
            for gene in genes_in_disease:
                if gene in common_genes:
                    net.add_edge(gene, disease)

    # Add Legend
    legend_nodes = {
        "Gene": "gray",
        "Protein": "gold",
        "Pathway": "skyblue",
        "Metabolite": "green",
        "Disease": "red"
    }
    for legend, color in legend_nodes.items():
        net.add_node(f"Legend_{legend}", label=legend, color=color, shape='box', physics=False)

    # Save and display network
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        with open(tmp_file.name, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=800)



# -----------------------------
# Grouped Association Table by Gene
# -----------------------------
if show_association_table and enrichment_results:
    st.subheader("🧬 Grouped Associations by Gene")

    grouped_data = {}

    # Initialize dictionary with genes
    for gene in common_genes:
        grouped_data[gene] = {
            "Gene": gene,
            "Proteins": set(),
            "Pathways": set(),
            "Metabolites": set(),
            "Diseases": set()
        }

    # Add proteins
    for _, row in pdf_filtered.iterrows():
        gene = row['Gene']
        protein = row['Protein']
        if gene in grouped_data:
            grouped_data[gene]["Proteins"].add(protein)

    # Add pathways
    if not reactome_df.empty:
        for _, row in reactome_df.iterrows():
            pathway = row['Term']
            for gene in row['Genes'].split(';'):
                if gene in grouped_data:
                    grouped_data[gene]["Pathways"].add(pathway)

    # Add metabolites
    if not hmdb_df.empty:
        for _, row in hmdb_df.iterrows():
            metabolite = row['Term']
            for gene in row['Genes'].split(';'):
                if gene in grouped_data:
                    grouped_data[gene]["Metabolites"].add(metabolite)

    # Add diseases
    if not disgenet_df.empty:
        for _, row in disgenet_df.iterrows():
            disease = row['Term']
            for gene in row['Genes'].split(';'):
                if gene in grouped_data:
                    grouped_data[gene]["Diseases"].add(disease)

    # Format for DataFrame
    grouped_list = []
    for gene, values in grouped_data.items():
        grouped_list.append({
            "Gene": gene,
            "Proteins": '; '.join(values["Proteins"]) if values["Proteins"] else '',
            "Pathways": '; '.join(values["Pathways"]) if values["Pathways"] else '',
            "Metabolites": '; '.join(values["Metabolites"]) if values["Metabolites"] else '',
            "Diseases": '; '.join(values["Diseases"]) if values["Diseases"] else '',
        })

    grouped_df = pd.DataFrame(grouped_list)
    st.dataframe(grouped_df)
