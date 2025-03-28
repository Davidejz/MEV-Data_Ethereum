# -*- coding: utf-8 -*-
"""scripts.ipynb
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import json

#Ethereum blocks Jan 2024: from block number 18908895 to 19129888

# Define file paths for CSV files
transactions_path = 'transactions_january_2024.csv'  # Ethereum transactions for January 2024
erc20_transfers_path = 'erc20_transfers_january_2024.csv'  # ERC-20 token transfers for January 2024

# Load CSV files into DataFrames
df_transactions = pd.read_csv(transactions_path)  # Transactions extracted using XBlock-ETH
df_erc20_transfers = pd.read_csv(erc20_transfers_path)  # ERC-20 token transfers extracted using XBlock-ETH

#To collect all internal transactions
def get_internal_transactions(tx_hash, api_key):
    url = f"https://api.etherscan.io/api?module=account&action=txlistinternal&txhash={tx_hash}&apikey={api_key}"

    try:
        response = requests.get(url)
        response.raise_for_status()

        # Parse the response
        data = response.json()

        return data['result']

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return []

    except json.JSONDecodeError:
        print("Error parsing the response as JSON.")
        return []

all_internal_transactions = []

api_key= 'API_KEY'

# Iterate over all transactionHash values in df_transactions
for tx_hash in df_transactions['transactionHash']:
    internal_transactions = get_internal_transactions(tx_hash, api_key)

    # Add the result (internal transactions) to the list
    all_internal_transactions.append(internal_transactions)

df_internal_transactions = pd.DataFrame(all_internal_transactions)
df_internal_transactions.to_csv('internal_transactions_january_2024.csv', index=False)

internal_transactions_path = 'internal_transactions_january_2024.csv'  # Internal transactions for January 2024 using Etherscan API
mempool_transactions_path = 'mempool_transactions_january_2024.csv'  # Pending transactions from Ethereum mempool Jan 2024 - mempool-dumpster

# Load CSV files into DataFrames
df_internal_transactions = pd.read_csv(internal_transactions_path)  # Internal transactions retrieved from Etherscan API
df_mempool_transactions = pd.read_csv(mempool_transactions_path)  # Mempool transactions collected via Mempool Dumpster

# Public or Private #
df_transactions['transaction_hash'] = df_transactions['transaction_hash'].str.lower()
df_mempool_transactions['transactionHash'] = df_mempool_transactions['transactionHash'].str.lower()

#Create the 'Type' column based on the presence of transaction_hash in df_mempool_transactions
df_transactions['Type'] = df_transactions['transaction_hash'].isin(df_mempool_transactions['transactionHash']).map({True: 'Public', False: 'Private'})

# MEV off-chain Data #
#MEV or NoMEV, that is, whether the transaction was included in a block originating from a relay within the MEV ecosystem.
#Specifically, this was determined by checking if the block hash matched any block hashes we collected via the APIs of the following relays: Aestus, Agnostic Gnosis, BloXroute MaxProfit, BloXroute Regulated, Eden Network, Flashbots, Manifold, Titan, and Ultrasound

