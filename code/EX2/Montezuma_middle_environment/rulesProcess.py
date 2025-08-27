import json
class Rule:
    def __init__(self,condition,action):
        self.condition = condition
        self.action = action

    def matches_condition(self,states):
        pass

    def execute_action(self):
        pass

def save_rules_to_file(rules,file_path='./rules.json'):
    with open(file_path,'w') as file:
        rulel_data = [{'condition':rule.condition,'action':rule.action} for rule in rules]
        json.dump(rulel_data,file)

def load_rules_from_file(file_path='./rules.json'):
    rules = []
    with open(file_path,'r') as file:
        rule_data = json.load(file)
        for rule_info in rule_data:
            rule = Rule(rule_info['condition'],rule_info['action'])
            rules.append(rule)
    return rules

