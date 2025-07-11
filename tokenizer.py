import torch
from collections import Counter
from itertools import chain
from transformers import AutoTokenizer
import json
import re

#prot_vocab = "ACDEFGHIKLMNPQRSTVWY"
#mol_vocab = "CNOHPSFIKLBcnohpsfikl1234567890#=@[]()/\\-+"

# TO DO: WHAT TO DO WITH THE SPECIAL CHARACTERS?? --> DECIDE WHAT TO DO WITH THEM
# left_characters = "Zr.eaVAugTRtWMdb<>*%:"
# The '\.' are ions that are not united with the molecule and I want to delete them
# The '\*' are Appendix/extremes/whatever of the molecule, so I will delete just the *
# The '%' means 2-digit ring number, so I want to preserve it (but maybe without separating the following 2-digit number)
# No idea of '<' and '>' , also I can not sketch these molecules
# The aromatic-bond symbol ’:’ can be used between aromatic atoms, but it is never necessary; a bond between two aromatic atoms is assumed to be aromatic unless it is explicitly represented as a single bond ’-’.
# The '\' and '/' means cis/trans configuration of double bonds, so I want to preserve them
# The '+' and '-' means charge, so I want to preserve it
# The 'Z' always appears as '[Zn]' which is a metal ion, so I think I will put all together
# Same happens with [Br] and [Cl] which are ions, na



class MolecularTokenizer:
    def __init__(self):
        self.cls_token = '<cls>'
        self.pad_token = '<pad>'
        self.eos_token = '<eos>'
        self.unk_token = '<unk>'
        self.special_tokens = [self.cls_token, self.pad_token, self.eos_token, self.unk_token]
        self.regex = '(Br?|Cl?|As|Sb|Se|Si|Sn|Te|H|N|O|S|P|F|I|b|c|n|o|s|p|\\(|\\)|\\[|\\]|\\.|\\=|\\#|\\-|\\+|\\\\|\\/|\\:|\\~|@@|@|\\?|>>?|\\*|\\$|\\%[0-9]{2}|[0-9])'
        self.build_vocab()
        
    def build_vocab(self):
        # Fixed vocabulary for molecular SMILES, 88 tokens
        chemical_tokens = [
            '#', '(', ')', '*', '+', '-', '/', '1', '2', '3', '4', '5', '6',\
            '7', '8', '9', '=', '@', '@@', 'As', 'B', 'Br', 'C', 'Cl', 'F',\
            'H', 'I', 'N', 'O', 'P', 'S', 'Sb', 'Se', 'Si', 'Sn', 'Te', '[',\
            '\\', ']', 'b' ,'c', 'n', 'o', 's'
            ]
        
        self.vocab = self.special_tokens + chemical_tokens
        self.vocab_size = len(self.vocab)

        # Create token2id and id2token mappings
        self.token2id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token = {idx: token for idx, token in enumerate(self.vocab)}

        # Ensure that special tokens are correctly mapped
        self.cls_token_id = self.token2id[self.cls_token]
        self.pad_token_id = self.token2id[self.pad_token]
        self.eos_token_id = self.token2id[self.eos_token]
        self.unk_token_id = self.token2id[self.unk_token]

    def __call__(self, molecule_list, truncation=True, padding=True, max_length=80):

        all_input_ids = []
        all_attention_masks = []

        if isinstance(molecule_list, str):
            molecule_list = [molecule_list]
        elif isinstance(molecule_list, list):
            pass
        else:
            raise TypeError('molecule_list must be either a single molecule \
                            or a list of molecules')

        longest = min(int(max(len(m) for m in molecule_list)), max_length)
        
        for molecule in molecule_list:
            
            regex = re.compile(self.regex)
            tokens = re.findall(regex, molecule)
            
            if truncation and len(tokens) > max_length - 2: # -2 to make space for cls and eos tokens
                tokens = tokens[:max_length - 2]
            
            tokens = [self.cls_token] + tokens + [self.eos_token]

            if padding and len(tokens) - 2 < longest:
                if max_length > longest:
                    tokens += [self.pad_token] * (longest - len(tokens) + 2)
                else:
                    tokens += [self.pad_token] * (longest - len(tokens))

            input_ids = [self.token2id.get(token, self.unk_token_id) for token in tokens]
            attention_mask = [0 if token != self.pad_token else 1 for token in tokens]

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)

        input_ids_tensor = torch.tensor(all_input_ids)
        attention_masks_tensor = torch.tensor(all_attention_masks)

        return {'input_ids': input_ids_tensor, 'attention_mask': attention_masks_tensor}

    def decode(self, token_ids, skip_special_tokens=False):
        
        try:
            tokens = [self.id2token[id] for id in token_ids]
        except:
            raise ValueError('tokenid not in MolecularTokenizer vocab')
        
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]

        return ''.join(tokens)

class ProteinTokenizer():
    def __init__(self):
        self.cls_token = '<cls>'
        self.pad_token = '<pad>'
        self.eos_token = '<eos>'
        self.unk_token = '<unk>'
        self.special_tokens = [self.cls_token, self.pad_token, self.eos_token, self.unk_token]
        self.build_vocab()

    def build_vocab(self):
        # Fixed vocabulary for proteins
        prot_vocab = "ACDEFGHIKLMNPQRSTVWY"
        self.vocab = self.special_tokens + list(prot_vocab)
        self.vocab_size = len(self.vocab)

        # Create token2id and id2token mappings
        self.token2id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token = {idx: token for idx, token in enumerate(self.vocab)}

        # Ensure that special tokens are correctly mapped
        self.cls_token_id = self.token2id[self.cls_token]
        self.pad_token_id = self.token2id[self.pad_token]
        self.eos_token_id = self.token2id[self.eos_token]
        self.unk_token_id = self.token2id[self.unk_token]

    def __call__(self, protein_list, truncation=True, padding=True, max_length=600):

        all_input_ids = []
        all_attention_masks = []

        if isinstance(protein_list, str):
            protein_list = [protein_list]
        elif isinstance(protein_list, list):
            pass
        else:
            raise TypeError('protein_list must be either a single protein \
                            or a list of proteins')

        longest = min(int(max(len(m) for m in protein_list)), max_length)

        for protein in protein_list:
            tokens = [x for x in protein]

            if truncation and len(tokens) > max_length - 2: # -2 to make space for cls and eos tokens
                tokens = tokens[:max_length - 2]

            tokens = [self.cls_token] + tokens + [self.eos_token]

            if padding and len(tokens) - 2 < longest:
                if max_length > longest:
                    tokens += [self.pad_token] * (longest - len(tokens) + 2)
                else:
                    tokens += [self.pad_token] * (longest - len(tokens))

            input_ids = [self.token2id.get(token, self.unk_token_id) for token in tokens]
            attention_mask = [0 if token != self.pad_token else 1 for token in tokens]

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)

        input_ids_tensor = torch.tensor(all_input_ids)
        attention_masks_tensor = torch.tensor(all_attention_masks)

        return {'input_ids': input_ids_tensor, 'attention_mask': attention_masks_tensor}
        
    def decode(self, token_ids, skip_special_tokens=False):
        
        try:
            tokens = [self.id2token[id] for id in token_ids]
        except:
            raise ValueError('tokenid not in ProteinTokenizer vocab')
        
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]

        return ''.join(tokens)

class Tokenizer:
    def __init__(self):
        self.delim_token = '<DELIM>'
        
        self.mol_tokenizer = MolecularTokenizer()
        self.prot_tokenizer = ProteinTokenizer()
        
        self.special_tokens = self.prot_tokenizer.special_tokens
        
        self.special_token_ids = [self.mol_tokenizer.cls_token_id,
                                  self.mol_tokenizer.pad_token_id,
                                  self.mol_tokenizer.eos_token_id,
                                  self.mol_tokenizer.unk_token_id]
        
        self.vocab_size = self.prot_tokenizer.vocab_size + self.mol_tokenizer.vocab_size\
                            - len(self.special_tokens) + 1 # +1 for the delimiter token
        
        self.build_vocab()

    def __call__(self, prots, mols, prot_max_length=600, mol_max_length=80):
        
        # Tokenize separately for protein, delimiter, and molecular sequences
        tokenized_prots = self.prot_tokenizer(prots, padding=True, truncation=True, max_length=prot_max_length)
        tokenized_mols = self. mol_tokenizer(mols, padding=True, truncation=True, max_length=mol_max_length)
        
        # Remove eos token id of prot input_ids and corresponding attention_mask
        eos_prot_id = self.prot_tokenizer.eos_token_id
        mask = tokenized_prots['input_ids'] != eos_prot_id
        prot_input_ids = [seq[mask[idx]] for idx, seq in enumerate(tokenized_prots['input_ids'])]
        prot_input_ids = torch.stack(prot_input_ids)
        prot_attention_mask = [seq[mask[idx]] for idx, seq in enumerate(tokenized_prots['attention_mask'])]
        prot_attention_mask = torch.stack(prot_attention_mask)
        
        # Remove cls token id of mol input_ids and corresponding attention_mask
        cls_mol_id = self.mol_tokenizer.cls_token_id
        mask = tokenized_mols['input_ids'] != cls_mol_id
        mols_input_ids = [seq[mask[idx]] for idx, seq in enumerate(tokenized_mols['input_ids'])]
        mols_input_ids = torch.stack(mols_input_ids)
        mols_attention_mask = [seq[mask[idx]] for idx, seq in enumerate(tokenized_mols['attention_mask'])]
        mols_attention_mask = torch.stack(mols_attention_mask)
        
        # Define delim_token_id based on the protein vocab size
        prot_vocab_size = self.prot_tokenizer.vocab_size
        delim_input_ids = (torch.tensor([self.delim_token_id] * len(prots))).unsqueeze(1)
        
        # Redefine molecular token ids to avoid duplicate token_ids between mols and prots
        mask = ~torch.isin(mols_input_ids, torch.tensor(self.special_token_ids))
        mols_input_ids[mask] += (prot_vocab_size + 1 - len(self.special_tokens))
        # not +2 because it is 0-indexed and without special tokens
        
        # Concatenate tokenized protein, delimiter, and molecular sequences
        input_tensor = torch.cat((prot_input_ids, delim_input_ids, mols_input_ids), dim=1)
        attention_mask = torch.cat((prot_attention_mask, torch.zeros_like(delim_input_ids),
                                    mols_attention_mask), dim=1)

        return {'input_ids': input_tensor, 'attention_mask': attention_mask}
    
    def build_vocab(self):
        
        self.delim_token_id = self.prot_tokenizer.vocab_size # not +1 because it is 0-indexed
        
        # update the id2token and token2id mappings
        updated_mol_token2id = {}
        for token, idx in self.mol_tokenizer.token2id.items():
            if token not in self.mol_tokenizer.special_tokens:
                updated_mol_token2id[token] = idx + self.prot_tokenizer.vocab_size + 1 - len(self.special_tokens)
            else:
                updated_mol_token2id[token] = idx
        
        updated_mol_id2token = {idx: token for token, idx in updated_mol_token2id.items()}

        # join the protein and the updated delim and molecular id2token mappings
        # Not token2id because the token is the same between the two tokenizers
        delim_id2token = {self.delim_token_id: self.delim_token}
        self.id2token = {**self.prot_tokenizer.id2token,**delim_id2token, **updated_mol_id2token}
        
        self.prot_ids = set(self.prot_tokenizer.id2token.keys())
        self.prot_ids = list(self.prot_ids - set(self.special_token_ids))
        self.mol_ids = set(updated_mol_id2token.keys())
        self.mol_ids = list(self.mol_ids - set(self.special_token_ids))
    
    def decode(self, token_ids, skip_special_tokens=True):
        
        try:
            tokens = [self.id2token[id] for id in token_ids]
        except:
            raise ValueError('tokenid not in Tokenizer vocab')
        
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
 
        return ''.join(tokens) 

if __name__ == '__main__':
    
    # Some examples here to test the tokenizers
    from data.fake_data import texts
    prot_list = [x.split('$')[0] for x in texts]
    molecule_list = [x.split('$')[-1] for x in texts]
    # print(prot_list, len(prot_list))
    # print(molecule_list, len(molecule_list))
    
    # Example usage of molecular tokenizer
    """
    molecular_tokenizer = MolecularTokenizer()
    encoded_molecule = molecular_tokenizer(molecule_list, truncation=True, padding=True, max_length=80)
    print(molecular_tokenizer.vocab)
    print(encoded_molecule['input_ids'])
    print(encoded_molecule['input_ids'][0])
    print(molecular_tokenizer.vocab_size)
    print(molecular_tokenizer.id2token)
    print(molecular_tokenizer.decode([2,  4,  4, 40, 36,  6, 41,  6,  4, 25, 36,  4,  4, 36,  4,  4, 36,  4, 25,  3,  0,  0,  0,  0,  0]))
    """

    # Example usage of protein tokenizer
    """
    prot_tokenizer = ProteinTokenizer()
    encoded_prots = prot_tokenizer(prot_list, padding=True, truncation=True, max_length=600)
    print(encoded_prots['input_ids'])
    print(prot_tokenizer.vocab_size)
    print(prot_tokenizer.build_vocab())
    """
    
    # Example usage of Tokenizer
    tokenizer = Tokenizer()
    encoded_texts = tokenizer(prot_list, molecule_list, prot_max_length=600, mol_max_length=50)
    input_tensor, attention_mask = encoded_texts['input_ids'], encoded_texts['attention_mask']

    print(tokenizer.vocab_size)
    print(prot_list[0], molecule_list[0])
    print(input_tensor[0])
    print(attention_mask[0])
    
    test_to_decode = [ 0, 14, 20,  7, 23, 12, 13, 21, 21, 21,  9,  4,  9,  9, 21,  9, 12, 19,
                        4, 13, 20, 11, 17, 13, 11, 17, 15, 10,  8, 21,  6,  7, 23,  6, 16, 20,
                        11,  7,  6, 19, 23, 18, 12, 17, 21, 21, 11,  6,  9,  7, 20, 21, 16, 14,
                        21, 13, 21,  9, 15,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 24, 25,
                        25, 61, 57, 27, 62, 27, 25, 46, 57, 25, 25, 57, 25, 25, 57, 25, 46,  2,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]
    
    decoded_text = tokenizer.decode(test_to_decode, skip_special_tokens=True)
    print(decoded_text)

