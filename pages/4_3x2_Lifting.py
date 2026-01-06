# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 02:47:39 2025

@author: xm7303
"""

# pages/4_3x2_Lifting.py
# Wrapper page for the 3×2 (R² → R³) SVD visualization

import streamlit as st
import app3x2  # assumes app3x2.py is in the project root (same folder as Home.py)

def main():
    # Optional: light page title here (app3x2.main() already sets its own title)
    st.write("")  # no-op, just keeps structure consistent
    app3x2.main()

if __name__ == "__main__":
    main()
