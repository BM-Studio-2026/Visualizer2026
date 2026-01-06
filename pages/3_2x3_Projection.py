# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 01:54:26 2025

@author: xm7303
"""

# pages/3_2x3_Projection.py

import streamlit as st
import app2x3  # assumes app2x3.py is in the project root

# Optional: set a page-specific title (no need for set_page_config here)
st.title("2×3 Projection: R³ → R² via SVD")

# Just delegate everything to app2x3
app2x3.main()
