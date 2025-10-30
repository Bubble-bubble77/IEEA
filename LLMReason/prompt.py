
def rel_if_equal_en(rel_dic, ent1, ent1_name):
    """Construct a prompt to identify entities of the same type"""
    # Build neighbor relationship descriptions
    neighbor_descriptions = []
    for neighbor, relation in rel_dic.items():
        neighbor_descriptions.append(f"- Relation '{relation}' connects to entity '{neighbor}'")
    
    neighbors_text = "\n".join(neighbor_descriptions)
    prompt = f"""
        Task: Analyze the given entity and its neighbor relationships to identify which neighbors 
        are of the same type as '{ent1_name}(ID: {ent1})'.

        Entity '{ent1_name}(ID: {ent1})' has the following neighbor relationships:
        {neighbors_text}

        Criteria for same-type entities:
        - They should share common attributes
        - Consider similarities in functionality, properties, and category

        Output requirements:
        - Return ONLY a list of neighbor entity names that are the same type as '{ent1}\t{ent1_name}'
        - Format: ["entity1", "entity2", ...]
        - If no matching entities found, return: []
        - Do NOT include any additional explanations or text

        Output example:
        ["example_entity1", "example_entity2"]
        or
        []
        """
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    return messages



def rel_if_equal_en_2(rel_dic, ent1, ent1_name):
    """Construct a prompt to identify entities of the same type"""
    neighbor_descriptions = []
    for neighbor, relation in rel_dic.items():
        neighbor_descriptions.append(f"- Relation '{relation}' connects to entity '{neighbor}'")
    
    neighbors_text = "\n".join(neighbor_descriptions)
    
    prompt = f"""
        Task: Identify neighbor entities that are the EXACT SAME TYPE as the target entity '{ent1}' ({ent1_name}).
        
        Target Entity Information:
        - Entity ID: '{ent1}'
        - Entity Name: '{ent1_name}'
        
        Neighbor Relationships:
        {neighbors_text}
        
        Same-Type Criteria (MUST meet ALL):
        1. Must belong to the same high-level category
        2. Must share core attributes 
        3. Must serve similar functions 
        4. Must be at the same level of abstraction 
        
        Important Clarifications:
        - "Same type" means identical entity type (e.g., Film to Film)
        - Focus ONLY on the entity type, not relationship semantics
        
        Output Requirements:
        - ONLY output a Python-formatted list of neighbor entity names
        - Format: ["entity_name1", "entity_name2", ...]
        - If NO matches: []
        - ABSOLUTELY NO additional text or explanations
        
        Examples of Valid Output:
        ["The_Dark_Knight", "Inception"]
        []
        """
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    return messages