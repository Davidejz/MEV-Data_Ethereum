# -*- coding: utf-8 -*-
"""
scripts.py
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import json
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

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

#Public or Private
df_transactions['transaction_hash'] = df_transactions['transaction_hash'].str.lower()
df_mempool_transactions['transactionHash'] = df_mempool_transactions['transactionHash'].str.lower()

#Create the 'Type' column based on the presence of transaction_hash in df_mempool_transactions
df_transactions['Type'] = df_transactions['transaction_hash'].isin(df_mempool_transactions['transactionHash']).map({True: 'Public', False: 'Private'})

#MEV off-chain Data
#MEV or NoMEV, that is, whether the transaction was included in a block originating from a relay within the MEV ecosystem.
#Specifically, this was determined by checking if the block hash matched any block hashes we collected via the APIs of the following relays: Aestus, Agnostic Gnosis, BloXroute MaxProfit, BloXroute Regulated, Eden Network, Flashbots, Manifold, Titan, and Ultrasound

#Flashbots
flashbots = pd.DataFrame()
cursor = 0 #insert the right value

while len(flashbots) < 10000:
    url = (
        f"https://0xac6e77dfe25ecd6110b8e780608cce0dab71fdd5ebea22a16c0205200f2f8e2e3ad3b71d3499c54ad14d6c21b41a37ae@boost-relay.flashbots.net/relay/v1/data/bidtraces/proposer_payload_delivered?cursor={cursor}"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data:
            continue

        new_data = pd.DataFrame.from_records(data)
        flashbots = pd.concat([flashbots, new_data], ignore_index=True)

        # Update cursor to the latest slot value
        cursor = int(flashbots.slot.iloc[-1])

    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching data at cursor {cursor}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

f= flashbots.drop_duplicates('slot')
f.reset_index(inplace=True, drop=True)
f.to_csv('flashbots_relay.csv')

#BloXroute MaxProfit
blxr_MaxProfit = pd.DataFrame()
cursor = 0

while len(blxr_MaxProfit) < 10000:
    url = (
        f"https://0x8b5d2e73e2a3a55c6c87b8b6eb92e0149a125c852751db1422fa951e42a09b82c142c3ea98d0d9930b056a3bc9896b8f@bloxroute.max-profit.blxrbdn.com/relay/v1/data/bidtraces/proposer_payload_delivered?cursor={cursor}"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data:
            continue

        new_data = pd.DataFrame.from_records(data)
        blxr_MaxProfit = pd.concat([blxr_MaxProfit, new_data], ignore_index=True)

        # Update cursor to the latest slot value
        cursor = int(blxr_MaxProfit.slot.iloc[-1])

    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching data at cursor {cursor}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

bmp= blxr_MaxProfit.drop_duplicates('slot')
bmp.reset_index(inplace=True, drop=True)
bmp.to_csv('blxr_MaxProfit_relay.csv')

#BloXroute Regulated
blxr_Regulated = pd.DataFrame()
cursor = 0

while len(blxr_Regulated) < 10000:
    url = (
        f"https://0xb0b07cd0abef743db4260b0ed50619cf6ad4d82064cb4fbec9d3ec530f7c5e6793d9f286c4e082c0244ffb9f2658fe88@bloxroute.regulated.blxrbdn.com/relay/v1/data/bidtraces/proposer_payload_delivered?cursor={cursor}"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data:
            continue

        new_data = pd.DataFrame.from_records(data)
        blxr_Regulated = pd.concat([blxr_Regulated, new_data], ignore_index=True)

        # Update cursor to the latest slot value
        cursor = int(blxr_Regulated.slot.iloc[-1])

    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching data at cursor {cursor}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

br= blxr_Regulated.drop_duplicates('slot')
br.reset_index(inplace=True, drop=True)
br.to_csv('blxr_Regulated_relay.csv')

#Eden Network
edenNetwork = pd.DataFrame()
cursor = 0

while len(edenNetwork) < 10000:
    url = (
        f"https://relay.edennetwork.io/relay/v1/data/bidtraces/proposer_payload_delivered?cursor={cursor}"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data:
            continue

        new_data = pd.DataFrame.from_records(data)
        edenNetwork = pd.concat([edenNetwork, new_data], ignore_index=True)

        # Update cursor to the latest slot value
        cursor = int(edenNetwork.slot.iloc[-1])

    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching data at cursor {cursor}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

e= edenNetwork.drop_duplicates('slot')
e.reset_index(inplace=True, drop=True)
e.to_csv('edenNetwork_relay.csv')

#Manifold / SecureRPC
securerpc = pd.DataFrame()
cursor = 0

while len(securerpc) < 10000:
    url = (
        f"https://mainnet-relay.securerpc.com/relay/v1/data/bidtraces/proposer_payload_delivered?cursor={cursor}"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data:
            continue

        new_data = pd.DataFrame.from_records(data)
        securerpc = pd.concat([securerpc, new_data], ignore_index=True)

        # Update cursor to the latest slot value
        cursor = int(securerpc.slot.iloc[-1])

    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching data at cursor {cursor}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

msrpc= securerpc.drop_duplicates('slot')
msrpc.reset_index(inplace=True, drop=True)
msrpc.to_csv('securerpc_relay.csv')

#Ultrasound
ultrasound = pd.DataFrame()
cursor = 0

while len(ultrasound) < 10000:
    url = (
        f"https://relay-analytics.ultrasound.money/relay/v1/data/bidtraces/proposer_payload_delivered?cursor={cursor}"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data:
            continue

        new_data = pd.DataFrame.from_records(data)
        ultrasound = pd.concat([ultrasound, new_data], ignore_index=True)

        # Update cursor to the latest slot value
        cursor = int(ultrasound.slot.iloc[-1])

    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching data at cursor {cursor}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

u= ultrasound.drop_duplicates('slot')
u.reset_index(inplace=True, drop=True)
u.to_csv('ultrasound_relay.csv')

#Aestus
aestus = pd.DataFrame()
cursor = 0

while len(aestus) < 10000:
    url = (
        f"https://mainnet.aestus.live/relay/v1/data/bidtraces/proposer_payload_delivered?cursor={cursor}"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data:
            continue

        new_data = pd.DataFrame.from_records(data)
        aestus = pd.concat([aestus, new_data], ignore_index=True)

        # Update cursor to the latest slot value
        cursor = int(aestus.slot.iloc[-1])

    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching data at cursor {cursor}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

ae= aestus.drop_duplicates('slot')
ae.reset_index(inplace=True, drop=True)
ae.to_csv('aestus_relay.csv')

#Agnostic
agnostic = pd.DataFrame()
cursor = 0

while len(agnostic) < 10000:
    url = (
        f"https://agnostic-relay.net/relay/v1/data/bidtraces/proposer_payload_delivered?cursor={cursor}"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data:
            continue

        new_data = pd.DataFrame.from_records(data)
        agnostic = pd.concat([agnostic, new_data], ignore_index=True)

        # Update cursor to the latest slot value
        cursor = int(agnostic.slot.iloc[-1])

    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching data at cursor {cursor}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

ag= agnostic.drop_duplicates('slot')
ag.reset_index(inplace=True, drop=True)
ag.to_csv('agnostic_relay.csv')

#Titan
titan = pd.DataFrame()
cursor = 0

while len(titan) < 10000:
    url = (
        f"https://0x8c4ed5e24fe5c6ae21018437bde147693f68cda427cd1122cf20819c30eda7ed74f72dece09bb313f2a1855595ab677d@titanrelay.xyz/relay/v1/data/bidtraces/proposer_payload_delivered?cursor={cursor}"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data:
            continue

        new_data = pd.DataFrame.from_records(data)
        titan = pd.concat([titan, new_data], ignore_index=True)

        # Update cursor to the latest slot value
        cursor = int(titan.slot.iloc[-1])

    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching data at cursor {cursor}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

t= titan.drop_duplicates('slot')
t.reset_index(inplace=True, drop=True)
t.to_csv('titan_relay.csv')

df_ae= pd.read_csv('aestus_relay.csv', index_col='Unnamed: 0')
df_ae['relay']= 'Aestus'
df_ag= pd.read_csv('agnostic_relay.csv', index_col='Unnamed: 0')
df_ag['relay']= 'Agnostic'
df_blxr_mx= pd.read_csv('blxr_MaxProfit_relay.csv', index_col='Unnamed: 0')
df_blxr_mx['relay']= 'BloXroute MaxProfit'
df_blxr_reg= pd.read_csv('blxr_Regulated_relay.csv', index_col='Unnamed: 0')
df_blxr_reg['relay']= 'BloXroute Regulated'
df_titan= pd.read_csv('titan_relay.csv', index_col='Unnamed: 0')
df_titan['relay']= 'Titan'
df_ultrasound= pd.read_csv('ultrasound_relay.csv', index_col='Unnamed: 0')
df_ultrasound['relay']= 'Ultrasound'
df_fl= pd.read_csv('flashbots_relay.csv', index_col='Unnamed: 0')
df_fl['relay']= 'Flashbots'
df_srpc= pd.read_csv('securerpc_relay.csv', index_col='Unnamed: 0')
df_srpc['relay']= 'Manifold'
df_eden= pd.read_csv('edenNetwork_relay.csv', index_col='Unnamed: 0')
df_eden['relay']= 'Eden'

df_mev= pd.concat([df_ae, df_ag, df_blxr_mx, df_blxr_reg, df_titan, df_ultrasound, df_fl, df_srpc, df_eden], axis=0)
df_mev.sort_values('slot', inplace=True)
df_mev.reset_index(inplace=True, drop=True)

#block Jan 2024
df_mev_filtered = df_mev[(df_mev['block_number'] >= 18908895) & (df_mev['block_number'] <= 19129888)]
df_mev_filtered.reset_index(inplace=True, drop=True)

bm= pd.read_csv('block_january_2024.csv', index_col='Unnamed: 0') # Blocks information extracted using XBlock-ETH
bh= pd.read_csv('block_hash_january_2024.csv', index_col='Unnamed: 0') # Blocks hash January 2024
bh.sort_values('number', inplace=True)
b = bm.merge(bh, left_on='blockNumber', right_on='number', how='inner') # Add the block hash to blocks information data
b_mev = b.merge(df_mev_filtered[['slot', 'parent_hash', 'block_hash', 'builder_pubkey', 'proposer_fee_recipient', 'gas_limit', 'gas_used', 'value', 'num_tx', 'block_number']], left_on='block_hash', right_on='block_hash', how='left')
b_mev.drop_duplicates(inplace=True)
b_mev.sort_values('blockNumber', inplace=True)
b_mev.drop_duplicates(subset='blockNumber', inplace=True)
b_mev = b_mev[b_mev['builder_pubkey'].notnull()]
b_mev.reset_index(inplace=True, drop=True)
b_mev.to_csv('mev_data.csv')

df_transactions['MEV'] = df_transactions['block_number'].isin(b_mev['blockNumber']).map({True: 'MEV', False: 'NoMEV'})

#Builder-payment
df_erc20_transfers = pd.read_csv('erc20_transfers_january_2024.csv')  # ERC-20 token transfers extracted using XBlock-ETH
b_mev= pd.read_csv('mev_data.csv', index_col='Unnamed: 0')

smart_contracts = set(df_erc20_transfers.loc[df_erc20_transfers['fromIsContract'] == 1, 'from']).union(df_erc20_transfers.loc[df_erc20_transfers['toIsContract'] == 1, 'to'])
smart_contracts_list = list(smart_contracts)
df_smart_contracts = pd.DataFrame({'smart_contract_address': smart_contracts_list})
df_smart_contracts.to_csv('smart_contracts.csv', index=False)
#add the other addresses that are smart contracts using the eth_getCode method of the MetaMask API

eoa = set(df_erc20_transfers.loc[df_erc20_transfers['fromIsContract'] == 0, 'from']).union(df_erc20_transfers.loc[df_erc20_transfers['toIsContract'] == 0, 'to'])
eoa = list(eoa)
df_eoa = pd.DataFrame({'eoa_address': smart_contracts_list})
df_eoa.to_csv('eoa.csv', index=False)
#add the other addresses that are smart contracts using the eth_getCode method of the MetaMask API

builder= b_mev['blockMiner'].unique()
validator= b_mev['proposer_fee_recipient'].unique()

tr = pd.read_csv('transactions_january_2024.csv')  # Transactions extracted using XBlock-ETH

all_addresses = set(tr['from_address']).union(set(tr['to_address']))
address_roles = []

for addr in all_addresses:
  if addr in builder and addr in validator:
    role = 'Builder Validator'
  elif addr in eoa:
    role = 'EOA'
  elif addr in smart_contracts:
    role = 'SmartContract'
  elif addr in builder:
    role = 'Builder'
  elif addr in validator:
    role = 'Validator'
  else:
    role = ''
    address_roles.append({'address': addr, 'role': role})

df_roles = pd.DataFrame(address_roles)
df_roles.to_csv('address_roles.csv', index=False)

g= tr[['transaction_hash','block_number','from_address','to_address']]
g.rename(columns={'from_address': 'source', 'to_address': 'target','transaction_hash':'TxHash'}, inplace=True)

internal_trx = pd.read_csv('internal_transactions_january_2024.csv')  # Internal transactions retrieved from Etherscan API
internal= internal_trx[['TxHash','From','To']]

smart_contract_addresses = df_roles[df_roles['role'] == 'Smart Contract']['address']
g_filtered = g[g['target'].isin(smart_contract_addresses)]

g_filtered['original'] = 1
merged = g_filtered.merge(internal, how='outer',
                  left_on=['TxHash', 'source', 'target'],
                  right_on=['TxHash', 'From', 'To'],
                  indicator=True)

merged.loc[merged['_merge'] == 'left_only', 'From'] = merged.loc[merged['_merge'] == 'left_only', 'source']
merged.loc[merged['_merge'] == 'left_only', 'To'] = merged.loc[merged['_merge'] == 'left_only', 'target']
merged.loc[merged['_merge'] == 'right_only', 'original'] = 0
merged.drop_duplicates(inplace=True)
merged.sort_values(by=['TxHash','original'],ascending=False,inplace=True)
merged = merged.dropna(subset=['To'])
g_targets= list(g_filtered.target.value_counts(ascending=True).index)
merged['From'] = merged['From'].str.lower()
merged['To'] = merged['To'].str.lower()
merged['is_source_contract'] = merged['From'].isin(smart_contract_addresses)
merged['is_target_contract'] = merged['To'].isin(smart_contract_addresses)
filtered_df = merged[merged['original'] == 1]
filtered_df = filtered_df[filtered_df['To'].isin(g_filtered['target'])]
filtered_df = filtered_df[filtered_df['is_target_contract'] == True]
merged = merged[merged['TxHash'].isin(filtered_df['TxHash'])]
g_filtered['is_target_contract'] = g_filtered['target'].isin(smart_contract_addresses)
g_filtered['original_sender'] = None
g_filtered.loc[g_filtered['is_target_contract'], 'original_sender'] = g['source']
g_filtered.rename(columns={'transaction_hash': 'TxHash'}, inplace=True)

df_with_original_sender = merged.merge(
    g_filtered[['TxHash', 'original_sender']],
    on='TxHash',
    how='left'
)

df_with_original_sender['From'] = df_with_original_sender['From'].str.lower()
df_with_original_sender['To'] = df_with_original_sender['To'].str.lower()
df_with_original_sender['original_sender'] = df_with_original_sender['original_sender'].str.lower()
df_with_original_sender.set_index('TxHash', inplace=True)


def process_row(index, row):
    transaction = row.copy()
    tx_hash = index
    original_sender = transaction['original_sender']

    if original_sender is not None:
        if transaction['is_source_contract']:
            transaction['From'] = original_sender
        if transaction['is_target_contract']:
            transaction['To'] = original_sender

    if transaction['From'] != transaction['To']:
        transaction['TxHash'] = tx_hash
        return transaction

    return None

processed_transactions = Parallel(n_jobs=-1)(
    delayed(process_row)(index, row) for index, row in tqdm(df_with_original_sender.iterrows(), total=df_with_original_sender.shape[0], desc="Processing transactions", position=0)
)

result_df = pd.DataFrame([tx for tx in processed_transactions if tx is not None])
result_df.reset_index(drop=True, inplace=True)
result_df.to_csv('TxHash_NoSM.csv')
result_df = result_df.merge(tr[['transaction_hash', 'block_number']], left_on='TxHash', right_on='transaction_hash', how='left')
result_df = result_df.merge(b_mev[['blockNumber', 'blockMiner', 'proposer_fee_recipient']], left_on='block_number', right_on='blockNumber', how='left')

df_with_roles = result_df.merge(df_roles[['address', 'role']], how='left', left_on='From', right_on='address') \
                 .rename(columns={'role': 'source_type'}) \
                 .drop(columns=['address'])

df_with_roles = df_with_roles.merge(df_roles[['address', 'role']], how='left', left_on='To', right_on='address') \
                           .rename(columns={'role': 'target_type'}) \
                           .drop(columns=['address'])

tx_hashes_builder = df_with_roles[
    (df_with_roles['original_sender'] == df_with_roles['From']) &
    (df_with_roles['target_type'].isin(['Builder', 'Builder Validator'])) &
    (df_with_roles['To'] == df_with_roles['blockMiner'])
]['TxHash'].tolist()

tx_hashes_builder.to_csv("tx_hash_builderPayment.csv")

df_transactions['BuilderPayment'] = df_transactions['transaction_hash'].isin(tx_hashes_builder['transaction_hash']).map({True: 'BuilderPayment', False: 'No'})

base_url = "https://data.zeromev.org/v1/mevBlock"

start_block = 18908895
end_block = 19129889
count = 100

data_list = []

current_block = start_block
with tqdm(total=(end_block - start_block) // count, desc="") as pbar:
    while current_block < end_block:
        try:
            response = requests.get(f"{base_url}?block_number={current_block}&count={count}")

            if response.status_code == 200:
                block_data = response.json()

                if block_data:
                    data_list.extend(block_data)
                    max_block = max(item['block_number'] for item in block_data)
                    current_block = max_block + 1
                else:
                    break
            else:
                print(f"Error {current_block}: {response.status_code}")
                break
            pbar.update(1)
            time.sleep(0.1)

        except Exception as e:
            print(f"Error {current_block}: {e}")
            break

df = pd.DataFrame(data_list)

merged_df = pd.merge(df, df_transactions,  how='left',
                     left_on=['block_number', 'address_from', 'address_to', 'tx_index'],
                     right_on=['block_number', 'from_address', 'to_address', 'tx_index'])

merged_df= merged_df.drop_duplicates()
merged_df.drop(columns=['from_address', 'to_address','arrival_time_us', 'arrival_time_eu', 'arrival_time_as'], inplace=True)
merged_df.to_csv("zeromev_data.csv", index=False)

df_transactions = df_transactions.merge(merged_df[['transaction_hash', 'mev_type', 'result']], on='transaction_hash', how='left')

#Tx_Index, Tx_Index_by_gas, Shift
df_transactions['Tx_Index'] = 0

for block_number, group in tqdm(df_transactions.groupby('block_number'), total=df_transactions['block_number'].nunique(), desc="Processing blocks"):
    df_transactions.loc[group.index, 'Tx_Index'] = range(1, len(group) + 1)

df_transactions = df_transactions.sort_values(by=['block_number', 'gas_price_gwei'], ascending=[True, False])


df_transactions['Tx_Index_by_gas'] = 0

for block_number, group in tqdm(df_transactions.groupby('block_number'), total=df_transactions['block_number'].nunique(), desc="Processing blocks by gas price"):
    df_transactions.loc[group.index, 'Tx_Index_by_gas'] = range(1, len(group) + 1)

df_transactions['shift'] = df_transactions['Tx_Index'] - df_transactions['Tx_Index_by_gas']

#Final dataframe
df_transactions
