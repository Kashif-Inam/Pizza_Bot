import os
import re
import time
import json
import spacy
import torch
import pickle
import numpy as np
import tensorflow_text
import tensorflow as tf
from copy import deepcopy
import tensorflow_hub as hub
from transformers import pipeline
from sklearn.preprocessing import LabelBinarizer

from IPython.display import clear_output
clear_output(wait=False)

app = Flask(__name__)

class clarity_chatbot():
    def __init__(self, intent_classifier_path, intent_labels, ner_model_path, dominos_db):
        # Load the intent classifier
        self.classifier_model = tf.keras.models.load_model(intent_classifier_path, custom_objects={'KerasLayer':hub.KerasLayer})
        with open(intent_labels, "rb") as f:
            self.binarizer = pickle.load(f)
        
        #Load the ner model
        self.ner = spacy.load(ner_model_path)
        
        #Load spacy model
        self.nlp = spacy.load("en_core_web_sm")
        
        #Dominos db
        self.dominos_db = dominos_db
        
        #Order details
        self.complete_order = []
        self.order_details = {"name":"", "delivery_type":"", "payment_method":"", "phone_number":"", "address":""}
        self.pizza_list = []
        self.side_item_list = []
        
        self.pizza_count_in_alpha = {0:"first", 1:"second", 2:"third", 3:"fourth", 4:"fifth", 5:"sixth"}
        
        #Conversation flow parameters
        self.intent = ""
        self.call_ended = False
        self.begin_pizza_order = False
        self.begin_side_order = False
        self.comp_pizza_order = False
        self.comp_side_order = True
        self.error_count = 0
        self.current_state = ""
        self.prev_response = ""
        self.selected_item = ""
        self.prev_intent = ""
        
    def clear_pizza_entries(self):
        self.intent = ""
        self.complete_order = []
        self.order_details = {"name":"", "delivery_type":"", "payment_method":"", "phone_number":"", "address":""}
        self.pizza_list = []
        self.side_item_list = []
        self.call_ended = False
        self.begin_pizza_order = False
        self.begin_side_order = False
        self.comp_pizza_order = False
        self.comp_side_order = True
        self.error_count = 0
        self.current_state = ""
        self.prev_response = ""
        self.selected_item = ""
        self.prev_intent = ""
        
    #Extract entities from a customer message
    def extract_entities(self, message):
        doc = self.ner(message)
        entities = []
        for ent in doc.ents:
            entities.append((ent.label_, ent.text))
        
        return entities

    #Classify the intent of a customer message
    def classify_intent(self, message):
        text = np.array(message).reshape(1, -1)
        prediction = tf.nn.softmax(self.classifier_model(text))
        predicted_class = self.binarizer.inverse_transform(prediction.numpy())[0]
        confidence = tf.reduce_max(prediction).numpy()
        
        return predicted_class,confidence
    
    #Check message is question or not
    def is_asking(self, sentence):
        doc = self.nlp(sentence)

        # Check if the sentence ends with a question mark
        if sentence.strip().endswith("?"):
            return True

        # Check for question indicators
        question_indicators = ["if", "tell", "wh", "how", "can", "could", "would", "is", "are", "was", "were", "do", "does", "did", "has", "have", "had", "should", "shall", "will", "may"]

        # Check if the sentence contains a question indicator with dependency "aux" or "ROOT" and is not a pronoun
        for token in doc:
            if token.lower_ in question_indicators and token.dep_ in ("aux", "ROOT") and token.pos_ != "PRON":
                return True

        return False
    
    #Check price in the db
    def check_price_in_db(self, return_price=False):
        comp_price = 0
        for pizza_list in self.pizza_list:
            if pizza_list["comp"]==True:
                size = pizza_list["pizza_size"]
                pizza_price = dominos_db[1]['price'][dominos_db[2]['size_id'][dominos_db[2]['size_name'].index(size)]-1] if dominos_db[0]['category_id'][0] == 1 else None
                extra_toppings = pizza_list["pizza_extra_toppings"]
                if type(extra_toppings)==list and len(extra_toppings):
                    size_id = {"small":1,"medium":2,"large":3,"extra large":4}[size]
                    topping_prices=[]
                    for topping in extra_toppings:
                        topping_prices.append([dominos_db[6]['topping_price'][i] for i in range(len(dominos_db[6]['topping_price'])) if dominos_db[6]['size_id'][i]==size_id and dominos_db[6]['topping_name'][i]==topping][0])
                    comp_price+=sum(topping_prices) + pizza_price
                else:
                    comp_price+=pizza_price
                pizza_list["price"] = (comp_price, f"{str(comp_price).split('.')[0]} dollars and {str(comp_price).split('.')[1]} cents")
        if return_price==True:
            total_price = 0
            for pizza_list in self.pizza_list:
                total_price+=pizza_list["price"][0]
            return f"{str(total_price).split('.')[0]} dollars and {str(total_price).split('.')[1][:2]} cents"
        
    #Check items in the db
    def check_items_in_db(self, menu=False, item="", category="", menu_item="", check_entity=""):
        def return_menu(category):
            if category=="size":
                return [size for size,category_id in zip(dominos_db[2]["size_name"],dominos_db[2]["category_id"]) if category_id==1]
            elif category=="flavor":
                return [item_name for item_name, cat in zip(self.dominos_db[3]["item_name"], self.dominos_db[3]["category_id"]) if cat == 1]
            elif category=="toppings" or category=="extra toppings":
                return [item_name for item_name in self.dominos_db[6]["topping_name"]]
            elif category=="crust":
                return [item_name for item_name in self.dominos_db[4]["crust_name"]]
            elif category=="sauce":
                return [item_name for item_name, cat in zip(self.dominos_db[5]["sauce_name"], self.dominos_db[5]["sauce_type"]) if cat == "pizza"]
        
        if item!="" and item!="no":
            if menu_item=="pizza":
                if category in ["quantity", "pizza"]:
                    return item
                pizza_items = return_menu(category)
                pizza_items.append("regular") if category in ["sauce","crust"] else pizza_items
                
                item = item.replace("crust","").replace("flavor","").replace("sauce","").strip()
                cor_item = [i for i in pizza_items if item in i]
                if len(cor_item):
                    cor_item = cor_item[0]
                    return cor_item
                else:
                    item = item.split(" ",1)[-1]
                    cor_item = [i for i in pizza_items if item in i]
                    if len(cor_item):
                        cor_item = cor_item[0]
                        return cor_item
                    else:
                        return False
        
        elif check_entity!="":
            pizza_size = [size for size,category_id in zip(dominos_db[2]["size_name"],dominos_db[2]["category_id"]) if category_id==1]
            pizza_flavor = [item_name for item_name, cat in zip(self.dominos_db[3]["item_name"], self.dominos_db[3]["category_id"]) if cat == 1]
            pizza_toppings = [item_name for item_name, cat in zip(self.dominos_db[3]["item_name"], self.dominos_db[3]["category_id"]) if cat == 1]
            pizza_crust = [item_name for item_name, cat in zip(self.dominos_db[3]["item_name"], self.dominos_db[3]["category_id"]) if cat == 1]
            pizza_sauce = [item_name for item_name, cat in zip(self.dominos_db[3]["item_name"], self.dominos_db[3]["category_id"]) if cat == 1]

            for name, list_ in zip(["pizza_size", "pizza_flavor", "pizza_toppings", "pizza_crust", "pizza_sauce"], [pizza_size, pizza_flavor, pizza_toppings, pizza_crust, pizza_sauce]):
                if check_entity in list_:
                    return name
            return ""
        
        if menu==True and category!="":
            all_items = return_menu(category)
            if category in ["quantity", "pizza"]:
                return True
            elif all_items is None:
                return ""
            else:
                all_items = list(set(all_items))
                return ",".join(i for i in all_items)
            
    
    #Check and return the next required entry for pizza
    def check_pizza_entry(self):
        for i,pizz_dict in enumerate(self.pizza_list):
            for key,value in pizz_dict.items():
                start_check=""
                if value!="deny" and key not in ["price","comp","start"]:
                    if pizz_dict["start"]==True:
                        start_check = f"Okay, Lets start with your {self.pizza_count_in_alpha[i]} pizza.\n" if len(self.pizza_list)>1 else "Okay. "
                    k = key.split("_",1)[-1].replace("_"," ")
                    self.current_state = key
                    if value=="" or value==[]:
                        pizz_dict["start"]=False
                        return start_check + f"What {k} would you like for the pizza" if len(self.pizza_list)>1 else start_check + f"What {k} would you like for the pizza"
                    elif value!="" or value!=[]:
                        if type(value)==list:
                            for n,item in enumerate(value):
                                cor_item = self.check_items_in_db(item=item, category=k, menu_item="pizza")
                                if cor_item!=False:
                                    pizz_dict[key][n] = cor_item
                                else:
                                    item_names = self.check_items_in_db(menu=True, category=k, menu_item="pizza")
                                    pizz_dict[key].remove(item)
                                    pizz_dict["start"]=False
                                    return start_check + f"The {item} you have told is not in our menu.\nPlease select it from these items {item_names}" if len(self.pizza_list)>1 else start_check + f"The {item} you have told is not in our menu.\nPlease select it from these items {item_names}"
                        elif type(value)==str:
                            cor_item = self.check_items_in_db(item=value, category=k, menu_item="pizza")
                            if cor_item!=False:
                                pizz_dict[key] = cor_item
                            else:
                                item_names = self.check_items_in_db(menu=True, category=k, menu_item="pizza")
                                pizz_dict["start"]=False
                                return start_check + f"The {k} you have told is not in our menu.\nPlease select it from these items {item_names}" if len(self.pizza_list)>1 else start_check + f"The {k} you have told is not in our menu.\nPlease select it from these items {item_names}"
                    else:
                        pizz_dict["start"]=False
                        return start_check + f"What {k} would you like for the pizza"  

            self.pizza_list[self.pizza_list.index(pizz_dict)]["comp"] = True
            self.check_price_in_db()
        if len(self.pizza_list):
            self.comp_pizza_order=True
        return True
    
    #Enter pizza entry
    def no_of_pizzas_entry(self, message, entities):
        if any('name' in i for i in entities):
            self.order_details["name"] = [i[1] for i in entities if i[0] == 'name']
        word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
        pizza_count = 0

        if len(self.selected_item):
            clear=False
            for n,i in enumerate(entities):
                if i[0]=="pizza_quantity":
                    entities.insert(n+1, self.selected_item[0])
                    clear=True
            self.selected_item="" if clear==True else self.selected_item

        if sum(1 for i in entities if i[0] == 'pizza_size')>sum(1 for i in entities if i[0] == 'pizza_quantity'):
            for i, e in enumerate(entities):
                if e[0]=='pizza_size':
                    entities.insert(i, ('pizza_quantity', 'one'))
                    break

        temp_dict = {"pizza_quantity": "1", "pizza_size": "", "pizza_flavor": "", "pizza_toppings": [], "pizza_crust": "", "pizza_sauce": "", "pizza_extra_toppings":[], "price":"", "comp": False, "start": True}

        for i, entity in enumerate(entities):
            self.begin_pizza_order = True
            if entity[0] == 'pizza_quantity':
                num = word_to_num.get(entity[1].lower(), None)
                if num is None:
                    return "redirect_to_agent"
                else:
                    pizza_count = num
            if "toppings" in entity[0]:
                temp_dict[entity[0]].append(entity[1])
            elif entity[0] in ["pizza_size"]:
                temp_dict[entity[0]] = entity[1]
            elif entity[0] in ["pizza_flavor", "pizza_sauce", "pizza_crust"]:
                temp_dict[entity[0]] = entity[1]

            if i == len(entities) - 1 or (i < len(entities) - 1 and entities[i+1][0] == "pizza_quantity"):
                if pizza_count==1:
                    for _ in range(pizza_count-1,pizza_count):
                        self.pizza_list.append(deepcopy(temp_dict))
                    pizza_count = 0
                    temp_dict = {"pizza_quantity": "1", "pizza_size": "", "pizza_flavor": "", "pizza_toppings": [], "pizza_crust": "", "pizza_sauce": "", "pizza_extra_toppings":[], "price":"", "comp": False, "start": True}

                elif pizza_count>1:
                    for _ in range(pizza_count,(pizza_count*2)):
                        self.pizza_list.append(deepcopy(temp_dict))
                    pizza_count = 0
                    temp_dict = {"pizza_quantity": "1", "pizza_size": "", "pizza_flavor": "", "pizza_toppings": [], "pizza_crust": "", "pizza_sauce": "", "pizza_extra_toppings":[], "price":"", "comp": False, "start": True}

        for pizz_dict in self.pizza_list:
            if "pizza_flavor" in pizz_dict.keys() and (pizz_dict["pizza_flavor"]!="" and "pizza_toppings" in pizz_dict.keys()):
                del pizz_dict["pizza_toppings"]
            elif "pizza_toppings" in pizz_dict.keys() and (pizz_dict["pizza_toppings"]!=[] and "pizza_flavor" in pizz_dict.keys()):
                del pizz_dict["pizza_flavor"]

        if not len(self.pizza_list):
            current_state = "pizza_quantity"
            return "How many pizzas would you like to order?"
        elif len(self.pizza_list)>6:
            return "redirect_to_agent"
        else:
            return self.check_pizza_entry()
    
    #Enter pizza specific required information
    def enter_pizza_info(self, message, entities, entity_name):
        for dict_index, pizz_dict in enumerate(self.pizza_list):
            if pizz_dict["comp"]==False:
                if message=="deny":
                    self.pizza_list[dict_index][self.current_state] = message
                else:
                    if len(entities):
                        for entity in entities:
                            if entity[0]!="menu_item":
                                if "toppings" in entity_name:
                                    self.pizza_list[dict_index][entity_name].append(entity[1])
                                else:
                                    if entity[0]==entity_name:
                                        self.pizza_list[dict_index][entity_name] = entity[1]
                    else:
                        return "Your voice has been broken.\n" + self.check_pizza_entry()
                
                if "pizza_flavor" in pizz_dict.keys() and (pizz_dict["pizza_flavor"]!="" and "pizza_toppings" in pizz_dict.keys()):
                    del self.pizza_list[dict_index]["pizza_toppings"]
                elif "pizza_toppings" in pizz_dict.keys() and (pizz_dict["pizza_toppings"]!=[] and "pizza_flavor" in self.pizza_list[dict_index].keys()):
                    del self.pizza_list[dict_index]["pizza_flavor"]
                
                break
        return self.check_pizza_entry()
                        
    #Order completed
    def order_completed(self):
        self.complete_order.append(self.pizza_list)
        
    #Print current order
    def print_current_order(self):
        order_text = ""
        for order_list in self.complete_order:
            for dict_ in order_list:
                for key,value in dict_.items():
                    if type(value)==list:
                        order_text = order_text + " " + " and ".join([i for i in value])
                    elif type(value)==str and value not in ["deny"]:
                        order_text = order_text + " " + value
                    else:
                        continue
                if order_list.index(dict_)!=(len(order_list)-1):
                    order_text = order_text + " and"
            if order_list!=[] and self.complete_order.index(order_list)!=(len(self.complete_order)-1):
                order_text = order_text# + " and side item of"
        order_text = order_text.strip() + " and your total is " + self.check_price_in_db(return_price=True)
        return order_text
    
    #Set conversation parameters
    def set_params(self):
        if len(self.pizza_list):
            self.comp_pizza_order=True
            for pizz_dict in self.pizza_list:
                if pizz_dict["comp"]==False:
                    self.comp_pizza_order=False
                    break
                
    def enter_order_details(self, entities):
        for entity in entities:
            if entity[0]=="order_type":
                self.order_details["delivery_type"] = entity[-1]
                
    #Correct the intent and entities from the db
    def correct_intent_entity(self, intent, confidence, org_entities, message):
        intent_dict = {"say_pizza_size":"pizza_size", "say_side_item":"", "ask_side_item":"", "order_pizza":"", "say_pizza_sauce":"pizza_sauce", "say_pizza_toppings":"pizza_toppings", "say_pizza_toppings":"pizza_extra_toppings", "say_pizza_crust":"pizza_crust", "say_pizza_quantity":"pizza_quantity", "say_pizza_flavor":"pizza_flavor"}
        
        org_entities = [entity for entity in org_entities if entity!=("name", "cali") and ("chicken" not in entity[1])] + [("pizza_flavor", "cali chicken bacon ranch")] if any(entity[0]=="name" and entity[1]=="cali" for entity in org_entities) and any("chicken" in entity[1] for entity in org_entities) else org_entities
        org_entities = [('pizza_flavor', 'chicken taco')] if org_entities==[('pizza_flavor', 'chicken'), ('menu_item', 'taco')] else org_entities
        org_entities = [("pizza_flavor", "wisconsin six cheese") if item[0]=="pizza_flavor" and item[1]=="cheese" else item for item in org_entities]
        org_entities = [("pizza_sauce", "robust tomato") if item[0]=="name" and item[1]=="tomato" else item for item in org_entities]
        org_entities = [("pizza_sauce", "alfredo sauce") if item[0]=="name" and item[1]=="alfredo" else item for item in org_entities]
        org_entities = [("pizza_toppings", "bacon") if item[0]=="order_type" and item[1]=="bacon" else item for item in org_entities]
        org_entities = [("pizza_toppings", "spinach") if item[0]=="side_stuffing" and item[1]=="spinach" else item for item in org_entities]
        org_entities = [("pizza_toppings", "feta") if item[0]=="side_stuffing" and item[1]=="feta" else item for item in org_entities]
        org_entities = [("pizza_toppings", "salami") if item[0]=="name" and item[1]=="salami" else item for item in org_entities]
        org_entities = [("pizza_toppings", org_entities[0][1])] if self.prev_intent=="ask_pizza_toppings" and len(org_entities)==1 and org_entities[0][0]=="pizza_flavor" and org_entities[0][1]=="pepperoni" else org_entities
        intent = "say_address" if self.current_state=="address" else intent
        
        if (self.current_state=="pizza_flavor" and self.prev_intent!="ask_pizza_toppings") or self.prev_intent in ["ask_pizza_flavor","ask_menu"]:
            intent = "say_pizza_flavor" if intent not in ["affirm", "deny"] else intent
            org_entities = [("pizza_flavor", "philly cheese steak") if item[0]=="name" and item[1]=="philly" else item for item in org_entities]
            org_entities = [("pizza_flavor", "memphis bbq chicken") if item[0]=="menu_item" and item[1]=="chicken" else item for item in org_entities]
        elif self.current_state in ["pizza_toppings","pizza_extra_toppings"] and self.prev_intent!="ask_pizza_flavor":
            intent = "say_pizza_toppings" if intent not in ["affirm", "deny"] else intent
            intent = "say_pizza_toppings" if (len(org_entities)==1 and org_entities[0][0]=="name") else intent
            org_entities = [("pizza_toppings", "philly steak") if item[0]=="name" and item[1]=="philly" else item for item in org_entities]
            org_entities = [("pizza_toppings", "salami") if item[0]=="name" and item[1]=="salami" else item for item in org_entities]
            org_entities = [("pizza_toppings", "cheddar") if item[0]=="name" and item[1]=="cheddar" else item for item in org_entities]
        
        if len(org_entities)==1 and org_entities[0][0] in ["pizza_sauce", "pizza_flavor"] and intent=="say_name":
            if org_entities[0][0]=="pizza_sauce":
                intent = "say_pizza_sauce"
            elif org_entities[0][0]=="pizza_flavor":
                intent = "say_pizza_flavor"
        
        if intent in list(intent_dict):
            entities = [i for i in org_entities if 'menu_item' not in i]
            entity_exists = set([i[0] for i in entities])
            if len(entity_exists)>1:
                if any(i for i in entities if "pizza" in i[0]):
                    return "order_pizza", entities
                elif all("side" in i[0] for i in entities):
                    return "say_side_item", entities
                else:
                    return "ask_again", entities
            elif len(entity_exists)==1:
                det_entity = ""
                for i,entity in enumerate(org_entities):
                    entity_name, entity_value = entity
                    if entity_name not in ["menu_item", "pizza_quantity"]:
                        det_entity = self.check_items_in_db(check_entity=entity_value)
                        if det_entity:
                            org_entities[i] = (det_entity,entity_value)
                
                if intent=="order_pizza" and org_entities[0][0]=="pizza_quantity":
                    return "say_pizza_quantity", org_entities
                if intent=="order_pizza" and org_entities[0][0]=="pizza_size" and self.current_state=="pizza_size":
                    return "say_pizza_size", org_entities
                if self.current_state=="pizza_sauce" and intent=="say_side_item":
                    return "say_pizza_sauce", org_entities
                if self.current_state=="pizza_sauce" and intent=="ask_side_item":
                    return "ask_pizza_sauce", org_entities
                if (self.current_state=="pizza_flavor" or self.prev_intent=="ask_pizza_flavor") and intent=="say_side_item":
                    return "say_pizza_flavor", org_entities
                if self.current_state=="pizza_flavor" and org_entities[0][0]=="pizza_flavor" and intent=="ask_side_item":
                    return "ask_pizza_flavor", org_entities
                if self.current_state=="pizza_flavor" and org_entities[0][0]=="pizza_toppings":
                    self.current_state = "pizza_toppings"
                    return "say_pizza_toppings", org_entities
                if self.current_state=="pizza_flavor" and org_entities[0][0]=="pizza_flavor":
                    self.current_state = "pizza_flavor"
                    return "say_pizza_flavor", org_entities
                if self.current_state=="pizza_sauce" and org_entities[0][0]=="pizza_sauce":
                    return "say_pizza_sauce", org_entities
                if self.current_state=="pizza_toppings" and org_entities[0][0]=="pizza_toppings":
                    self.current_state = "pizza_toppings"
                    return "say_pizza_toppings", org_entities
                if self.current_state==intent_dict[intent]:
                    return intent, org_entities
                if self.current_state=="" and org_entities[0][0]=="pizza_quantity":
                    return "say_pizza_quantity", org_entities
                if self.current_state=="" and "pizza" in intent:
                    return intent, org_entities
                if self.current_state!=intent_dict[intent]:
                    return "ask_again",[]
                else:
                    return False, []
            else:
                if len(org_entities)==1 and org_entities[0][0]=="menu_item" and org_entities[0][1]=="pizza":
                    return "order_pizza", org_entities
                if self.current_state=="pizza_sauce" and intent=="ask_side_item":
                    return "ask_pizza_sauce", org_entities
                else:
                    return intent, org_entities
        else:
            if self.current_state=="pizza_sauce" and intent=="ask_side_item":
                return "ask_pizza_sauce", org_entities
            elif intent in ["ask_pizza_flavor","ask_pizza_toppings"]:
                if "flavor" in message and "topping" not in message:
                    return "ask_pizza_flavor", org_entities
                elif "topping" in message and "flavor" not in message:
                    return "ask_pizza_toppings", org_entities
            else:
                return intent, org_entities
        
    
    #Find the entity from the db for the intent
    def find_entity_for_intent(self, message, prev_intent, entities):
        found_entity=[]
        non_pizz_intent_list = [("say_delivery_type", "order_type")]
        say_intent_list = [("say_pizza_size", "pizza_size"), ("say_pizza_sauce", "pizza_sauce"), ("say_sauce", "dipping_sauce"), ("say_pizza_toppings", "pizza_toppings"), ("say_pizza_toppings", "pizza_extra_toppings"), ("say_pizza_crust", "pizza_crust"), ("say_pizza_flavor", "pizza_flavor")]
        ask_intent_list = [("ask_pizza_size", "pizza_size"), ("ask_pizza_sauce", "pizza_sauce"), ("ask_sauce", "dipping_sauce"), ("ask_pizza_toppings", "pizza_toppings"), ("ask_pizza_toppings", "pizza_extra_toppings"), ("ask_pizza_crust", "pizza_crust"), ("ask_pizza_flavor", "pizza_flavor")]

        check_entities = [i for i in entities if 'menu_item' not in i]
        entity_exists = set([i[0] for i in check_entities])
        if len(entity_exists)>1:
            if self.current_state=="pizza_flavor":
                if "pizza_toppings" in entity_exists:
                    prev_intent = "say_pizza_toppings"
                    pass
                else:
                    return "order_pizza", entities
            else:
                return "order_pizza", entities
        
        if prev_intent=="say_pizza_quantity" or self.current_state in ["pizza_quantity"]:
class clarity_chatbot():
    def __init__(self, intent_classifier_path, intent_labels, ner_model_path, dominos_db):
        # Load the intent classifier
        self.classifier_model = tf.keras.models.load_model(intent_classifier_path, custom_objects={'KerasLayer':hub.KerasLayer})
        with open(intent_labels, "rb") as f:
            self.binarizer = pickle.load(f)
        
        #Load the ner model
        self.ner = spacy.load(ner_model_path)
        
        #Load spacy model
        self.nlp = spacy.load("en_core_web_sm")
        
        #Dominos db
        self.dominos_db = dominos_db
        
        #Order details
        self.complete_order = []
        self.order_details = {"name":"", "delivery_type":"", "payment_method":"", "phone_number":"", "address":""}
        self.pizza_list = []
        self.side_item_list = []
        
        self.pizza_count_in_alpha = {0:"first", 1:"second", 2:"third", 3:"fourth", 4:"fifth", 5:"sixth"}
        
        #Conversation flow parameters
        self.intent = ""
        self.call_ended = False
        self.begin_pizza_order = False
        self.begin_side_order = False
        self.comp_pizza_order = False
        self.comp_side_order = True
        self.error_count = 0
        self.current_state = ""
        self.prev_response = ""
        self.selected_item = ""
        self.prev_intent = ""
        
        #Save conversation for testing perpose
        count = 0
        self.conversation_file = ""
        
    def clear_pizza_entries(self):
        self.intent = ""
        self.complete_order = []
        self.order_details = {"name":"", "delivery_type":"", "payment_method":"", "phone_number":"", "address":""}
        self.pizza_list = []
        self.side_item_list = []
        self.call_ended = False
        self.begin_pizza_order = False
        self.begin_side_order = False
        self.comp_pizza_order = False
        self.comp_side_order = True
        self.error_count = 0
        self.current_state = ""
        self.prev_response = ""
        self.selected_item = ""
        self.prev_intent = ""
        
    #Extract entities from a customer message
    def extract_entities(self, message):
        doc = self.ner(message)
        entities = []
        for ent in doc.ents:
            entities.append((ent.label_, ent.text))
        
        return entities

    #Classify the intent of a customer message
    def classify_intent(self, message):
        text = np.array(message).reshape(1, -1)
        prediction = tf.nn.softmax(self.classifier_model(text))
        predicted_class = self.binarizer.inverse_transform(prediction.numpy())[0]
        confidence = tf.reduce_max(prediction).numpy()
        
        return predicted_class,confidence
    
    #Check message is question or not
    def is_asking(self, sentence):
        doc = self.nlp(sentence)

        # Check if the sentence ends with a question mark
        if sentence.strip().endswith("?"):
            return True

        # Check for question indicators
        question_indicators = ["if", "tell", "wh", "how", "can", "could", "would", "is", "are", "was", "were", "do", "does", "did", "has", "have", "had", "should", "shall", "will", "may"]

        # Check if the sentence contains a question indicator with dependency "aux" or "ROOT" and is not a pronoun
        for token in doc:
            if token.lower_ in question_indicators and token.dep_ in ("aux", "ROOT") and token.pos_ != "PRON":
                return True

        return False
    
    #Check price in the db
    def check_price_in_db(self, return_price=False):
        comp_price = 0
        for pizza_list in self.pizza_list:
            if pizza_list["comp"]==True:
                size = pizza_list["pizza_size"]
                pizza_price = dominos_db[1]['price'][dominos_db[2]['size_id'][dominos_db[2]['size_name'].index(size)]-1] if dominos_db[0]['category_id'][0] == 1 else None
                extra_toppings = pizza_list["pizza_extra_toppings"]
                if type(extra_toppings)==list and len(extra_toppings):
                    size_id = {"small":1,"medium":2,"large":3,"extra large":4}[size]
                    topping_prices=[]
                    for topping in extra_toppings:
                        topping_prices.append([dominos_db[6]['topping_price'][i] for i in range(len(dominos_db[6]['topping_price'])) if dominos_db[6]['size_id'][i]==size_id and dominos_db[6]['topping_name'][i]==topping][0])
                    comp_price+=sum(topping_prices) + pizza_price
                else:
                    comp_price+=pizza_price
                pizza_list["price"] = [comp_price, f"{str(comp_price).split('.')[0]} dollars and {str(comp_price).split('.')[1]} cents"]
        if return_price==True:
            total_price = 0
            for pizza_list in self.pizza_list:
                total_price+=pizza_list["price"][0]
            return f"{str(total_price).split('.')[0]} dollars and {str(total_price).split('.')[1][:2]} cents"
        
    #Check items in the db
    def check_items_in_db(self, menu=False, item="", category="", menu_item="", check_entity=""):
        def return_menu(category):
            if category=="size":
                return [size for size,category_id in zip(dominos_db[2]["size_name"],dominos_db[2]["category_id"]) if category_id==1]
            elif category=="flavor":
                return [item_name for item_name, cat in zip(self.dominos_db[3]["item_name"], self.dominos_db[3]["category_id"]) if cat == 1]
            elif category=="toppings" or category=="extra toppings":
                return [item_name for item_name in self.dominos_db[6]["topping_name"]]
            elif category=="crust":
                return [item_name for item_name in self.dominos_db[4]["crust_name"]]
            elif category=="sauce":
                return [item_name for item_name, cat in zip(self.dominos_db[5]["sauce_name"], self.dominos_db[5]["sauce_type"]) if cat == "pizza"]
        
        if item!="" and item!="no":
            if menu_item=="pizza":
                if category in ["quantity", "pizza"]:
                    return item
                pizza_items = return_menu(category)
                pizza_items.append("regular") if category in ["sauce","crust"] else pizza_items
                
                item = item.replace("crust","").replace("flavor","").replace("sauce","").strip()
                cor_item = [i for i in pizza_items if item in i]
                if len(cor_item):
                    cor_item = cor_item[0]
                    return cor_item
                else:
                    item = item.split(" ",1)[-1]
                    cor_item = [i for i in pizza_items if item in i]
                    if len(cor_item):
                        cor_item = cor_item[0]
                        return cor_item
                    else:
                        return False
        
        elif check_entity!="":
            pizza_size = [size for size,category_id in zip(dominos_db[2]["size_name"],dominos_db[2]["category_id"]) if category_id==1]
            pizza_flavor = [item_name for item_name, cat in zip(self.dominos_db[3]["item_name"], self.dominos_db[3]["category_id"]) if cat == 1]
            pizza_toppings = [item_name for item_name, cat in zip(self.dominos_db[3]["item_name"], self.dominos_db[3]["category_id"]) if cat == 1]
            pizza_crust = [item_name for item_name, cat in zip(self.dominos_db[3]["item_name"], self.dominos_db[3]["category_id"]) if cat == 1]
            pizza_sauce = [item_name for item_name, cat in zip(self.dominos_db[3]["item_name"], self.dominos_db[3]["category_id"]) if cat == 1]

            for name, list_ in zip(["pizza_size", "pizza_flavor", "pizza_toppings", "pizza_crust", "pizza_sauce"], [pizza_size, pizza_flavor, pizza_toppings, pizza_crust, pizza_sauce]):
                if check_entity in list_:
                    return name
            return ""
        
        if menu==True and category!="":
            all_items = return_menu(category)
            if category in ["quantity", "pizza"]:
                return True
            elif all_items is None:
                return ""
            else:
                all_items = list(set(all_items))
                return ",".join(i for i in all_items)
            
    
    #Check and return the next required entry for pizza
    def check_pizza_entry(self):
        for i,pizz_dict in enumerate(self.pizza_list):
            for key,value in pizz_dict.items():
                start_check=""
                if value!="deny" and key not in ["price","comp","start"]:
                    if pizz_dict["start"]==True:
                        start_check = self.handle_conversational_response(condition="ask_for_more_pizza", no_of_pizza=i) if len(self.pizza_list)>1 else self.handle_conversational_response(condition="ask_for_one_pizza")
                    k = key.split("_",1)[-1].replace("_"," ")
                    self.current_state = key
                    if value=="" or value==[]:
                        pizz_dict["start"]=False
                        return start_check + self.handle_conversational_response(item=k, condition="ask_pizza_ingredient")
                    elif value!="" or value!=[]:
                        if type(value)==list:
                            for n,item in enumerate(value):
                                cor_item = self.check_items_in_db(item=item, category=k, menu_item="pizza")
                                if cor_item!=False:
                                    pizz_dict[key][n] = cor_item
                                else:
                                    item_names = self.check_items_in_db(menu=True, category=k, menu_item="pizza")
                                    pizz_dict[key].remove(item)
                                    pizz_dict["start"]=False
                                    return start_check + self.handle_conversational_response(item=item_names, condition="ask_pizza_ingredient", category=item)
                        elif type(value)==str:
                            cor_item = self.check_items_in_db(item=value, category=k, menu_item="pizza")
                            if cor_item!=False:
                                pizz_dict[key] = cor_item
                            else:
                                item_names = self.check_items_in_db(menu=True, category=k, menu_item="pizza")
                                pizz_dict["start"]=False
                                return start_check + self.handle_conversational_response(item=item_names, condition="ask_pizza_ingredient", category=k)
                    else:
                        pizz_dict["start"]=False
                        return start_check + self.handle_conversational_response(item=k, condition="ask_pizza_ingredient")

            self.pizza_list[self.pizza_list.index(pizz_dict)]["comp"] = True
            self.check_price_in_db()
        if len(self.pizza_list):
            self.comp_pizza_order=True
        return True
    
    #Enter pizza entry
    def no_of_pizzas_entry(self, message, entities):
        if any('name' in i for i in entities):
            self.order_details["name"] = [i[1] for i in entities if i[0] == 'name']
        word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
        pizza_count = 0

        if len(self.selected_item):
            clear=False
            for n,i in enumerate(entities):
                if i[0]=="pizza_quantity":
                    entities.insert(n+1, self.selected_item[0])
                    clear=True
            self.selected_item="" if clear==True else self.selected_item

        if sum(1 for i in entities if i[0] == 'pizza_size')>sum(1 for i in entities if i[0] == 'pizza_quantity'):
            for i, e in enumerate(entities):
                if e[0]=='pizza_size':
                    entities.insert(i, ('pizza_quantity', 'one'))
                    break

        temp_dict = {"pizza_quantity": "1", "pizza_size": "", "pizza_flavor": "", "pizza_toppings": [], "pizza_crust": "", "pizza_sauce": "", "pizza_extra_toppings":[], "price":"", "comp": False, "start": True}

        for i, entity in enumerate(entities):
            self.begin_pizza_order = True
            if entity[0] == 'pizza_quantity':
                num = word_to_num.get(entity[1].lower(), None)
                if num is None:
                    return "redirect_to_agent"
                else:
                    pizza_count = num
            if "toppings" in entity[0]:
                temp_dict[entity[0]].append(entity[1])
            elif entity[0] in ["pizza_size"]:
                temp_dict[entity[0]] = entity[1]
            elif entity[0] in ["pizza_flavor", "pizza_sauce", "pizza_crust"]:
                temp_dict[entity[0]] = entity[1]

            if i == len(entities) - 1 or (i < len(entities) - 1 and entities[i+1][0] == "pizza_quantity"):
                if pizza_count==1:
                    for _ in range(pizza_count-1,pizza_count):
                        self.pizza_list.append(deepcopy(temp_dict))
                    pizza_count = 0
                    temp_dict = {"pizza_quantity": "1", "pizza_size": "", "pizza_flavor": "", "pizza_toppings": [], "pizza_crust": "", "pizza_sauce": "", "pizza_extra_toppings":[], "price":"", "comp": False, "start": True}

                elif pizza_count>1:
                    for _ in range(pizza_count,(pizza_count*2)):
                        self.pizza_list.append(deepcopy(temp_dict))
                    pizza_count = 0
                    temp_dict = {"pizza_quantity": "1", "pizza_size": "", "pizza_flavor": "", "pizza_toppings": [], "pizza_crust": "", "pizza_sauce": "", "pizza_extra_toppings":[], "price":"", "comp": False, "start": True}

        for pizz_dict in self.pizza_list:
            if "pizza_flavor" in pizz_dict.keys() and (pizz_dict["pizza_flavor"]!="" and "pizza_toppings" in pizz_dict.keys()):
                del pizz_dict["pizza_toppings"]
            elif "pizza_toppings" in pizz_dict.keys() and (pizz_dict["pizza_toppings"]!=[] and "pizza_flavor" in pizz_dict.keys()):
                del pizz_dict["pizza_flavor"]

        if not len(self.pizza_list):
            current_state = "pizza_quantity"
            return self.handle_conversational_response(condition="ask_quantity")
        elif len(self.pizza_list)>6:
            return "redirect_to_agent"
        else:
            return self.check_pizza_entry()
    
    #Enter pizza specific required information
    def enter_pizza_info(self, message, entities, entity_name):
        for dict_index, pizz_dict in enumerate(self.pizza_list):
            if pizz_dict["comp"]==False:
                if message=="deny":
                    self.pizza_list[dict_index][self.current_state] = message
                else:
                    if len(entities):
                        for entity in entities:
                            if entity[0]!="menu_item":
                                if "toppings" in entity_name:
                                    self.pizza_list[dict_index][entity_name].append(entity[1])
                                else:
                                    if entity[0]==entity_name:
                                        self.pizza_list[dict_index][entity_name] = entity[1]
                    else:
                        return self.handle_error_response(part="first")+self.check_pizza_entry() if len(self.check_pizza_entry()) else self.handle_error_response(part="both")
                
                if "pizza_flavor" in pizz_dict.keys() and (pizz_dict["pizza_flavor"]!="" and "pizza_toppings" in pizz_dict.keys()):
                    del self.pizza_list[dict_index]["pizza_toppings"]
                elif "pizza_toppings" in pizz_dict.keys() and (pizz_dict["pizza_toppings"]!=[] and "pizza_flavor" in self.pizza_list[dict_index].keys()):
                    del self.pizza_list[dict_index]["pizza_flavor"]
                
                break
        return self.check_pizza_entry()
                        
    #Order completed
    def order_completed(self):
        self.complete_order.append(self.pizza_list)
        
    #Print current order
    def print_current_order(self):
        order_text = ""
        for order_list in self.complete_order:
            for dict_ in order_list:
                for key,value in dict_.items():
                    if key!="price":
                        if type(value)==list:
                            order_text = order_text + " " + " and ".join([i for i in value])
                        elif type(value)==str and value not in ["deny"]:
                            order_text = order_text + " " + value
                        else:
                            continue
                if order_list.index(dict_)!=(len(order_list)-1):
                    order_text = order_text + " and"
            if order_list!=[] and self.complete_order.index(order_list)!=(len(self.complete_order)-1):
                order_text = order_text# + " and side item of"
        order_text = order_text.strip() + " and your total is " + self.check_price_in_db(return_price=True)
        return order_text
    
    #Set conversation parameters
    def set_params(self):
        if len(self.pizza_list):
            self.comp_pizza_order=True
            for pizz_dict in self.pizza_list:
                if pizz_dict["comp"]==False:
                    self.comp_pizza_order=False
                    break
                
    def enter_order_details(self, entities):
        for entity in entities:
            if entity[0]=="order_type":
                self.order_details["delivery_type"] = entity[-1]
                
    #Correct the intent and entities from the db
    def correct_intent_entity(self, intent, confidence, org_entities, message):
        intent_dict = {"say_pizza_size":"pizza_size", "say_side_item":"", "ask_side_item":"", "order_pizza":"", "say_pizza_sauce":"pizza_sauce", "say_pizza_toppings":"pizza_toppings", "say_pizza_toppings":"pizza_extra_toppings", "say_pizza_crust":"pizza_crust", "say_pizza_quantity":"pizza_quantity", "say_pizza_flavor":"pizza_flavor"}
        
        org_entities = [entity for entity in org_entities if entity!=("name", "cali") and ("chicken" not in entity[1])] + [("pizza_flavor", "cali chicken bacon ranch")] if any(entity[0]=="name" and entity[1]=="cali" for entity in org_entities) and any("chicken" in entity[1] for entity in org_entities) else org_entities
        org_entities = [('pizza_flavor', 'chicken taco')] if org_entities==[('pizza_flavor', 'chicken'), ('menu_item', 'taco')] else org_entities
        org_entities = [("pizza_flavor", "wisconsin six cheese") if item[0]=="pizza_flavor" and item[1]=="cheese" else item for item in org_entities]
        org_entities = [("pizza_sauce", "robust tomato") if item[0]=="name" and item[1]=="tomato" else item for item in org_entities]
        org_entities = [("pizza_sauce", "alfredo sauce") if item[0]=="name" and item[1]=="alfredo" else item for item in org_entities]
        org_entities = [("pizza_toppings", "bacon") if item[0]=="order_type" and item[1]=="bacon" else item for item in org_entities]
        org_entities = [("pizza_toppings", "spinach") if item[0]=="side_stuffing" and item[1]=="spinach" else item for item in org_entities]
        org_entities = [("pizza_toppings", "feta") if item[0]=="side_stuffing" and item[1]=="feta" else item for item in org_entities]
        org_entities = [("pizza_toppings", "salami") if item[0]=="name" and item[1]=="salami" else item for item in org_entities]
        org_entities = [("pizza_toppings", org_entities[0][1])] if self.prev_intent=="ask_pizza_toppings" and len(org_entities)==1 and org_entities[0][0]=="pizza_flavor" and org_entities[0][1]=="pepperoni" else org_entities
        intent = "say_address" if self.current_state=="address" else intent
        
        if (self.current_state=="pizza_flavor" and self.prev_intent!="ask_pizza_toppings") or self.prev_intent in ["ask_pizza_flavor","ask_menu"]:
            intent = "say_pizza_flavor" if intent in ["say_name","say_side_item","ask_side_item"] else intent
            org_entities = [("pizza_flavor", "philly cheese steak") if item[0]=="name" and item[1]=="philly" else item for item in org_entities]
            org_entities = [("pizza_flavor", "memphis bbq chicken") if item[0]=="menu_item" and item[1]=="chicken" else item for item in org_entities]
        elif self.current_state in ["pizza_toppings","pizza_extra_toppings"] and self.prev_intent!="ask_pizza_flavor":
            intent = "say_pizza_toppings" if intent in ["say_name","say_side_item","ask_side_item"] else intent
            intent = "say_pizza_toppings" if (len(org_entities)==1 and org_entities[0][0]=="name") else intent
            org_entities = [("pizza_toppings", "philly steak") if item[0]=="name" and item[1]=="philly" else item for item in org_entities]
            org_entities = [("pizza_toppings", "salami") if item[0]=="name" and item[1]=="salami" else item for item in org_entities]
            org_entities = [("pizza_toppings", "cheddar") if item[0]=="name" and item[1]=="cheddar" else item for item in org_entities]
        
        if len(org_entities)==1 and org_entities[0][0] in ["pizza_sauce", "pizza_flavor"] and intent=="say_name":
            if org_entities[0][0]=="pizza_sauce":
                intent = "say_pizza_sauce"
            elif org_entities[0][0]=="pizza_flavor":
                intent = "say_pizza_flavor"
        
        if intent in list(intent_dict):
            entities = [i for i in org_entities if 'menu_item' not in i]
            entity_exists = set([i[0] for i in entities])
            if len(entity_exists)>1:
                if any(i for i in entities if "pizza" in i[0]):
                    return "order_pizza", entities
                elif all("side" in i[0] for i in entities):
                    return "say_side_item", entities
                else:
                    return "ask_again", entities
            elif len(entity_exists)==1:
                det_entity = ""
                for i,entity in enumerate(org_entities):
                    entity_name, entity_value = entity
                    if entity_name not in ["menu_item", "pizza_quantity"]:
                        det_entity = self.check_items_in_db(check_entity=entity_value)
                        if det_entity:
                            org_entities[i] = (det_entity,entity_value)
                
                if intent=="order_pizza" and org_entities[0][0]=="pizza_quantity":
                    return "say_pizza_quantity", org_entities
                if intent=="order_pizza" and org_entities[0][0]=="pizza_size" and self.current_state=="pizza_size":
                    return "say_pizza_size", org_entities
                if self.current_state=="pizza_sauce" and intent=="say_side_item":
                    return "say_pizza_sauce", org_entities
                if self.current_state=="pizza_sauce" and intent=="ask_side_item":
                    return "ask_pizza_sauce", org_entities
                if (self.current_state=="pizza_flavor" or self.prev_intent=="ask_pizza_flavor") and intent=="say_side_item":
                    return "say_pizza_flavor", org_entities
                if self.current_state=="pizza_flavor" and org_entities[0][0]=="pizza_flavor" and intent=="ask_side_item":
                    return "ask_pizza_flavor", org_entities
                if self.current_state=="pizza_flavor" and org_entities[0][0]=="pizza_toppings":
                    self.current_state = "pizza_toppings"
                    return "say_pizza_toppings", org_entities
                if self.current_state=="pizza_flavor" and org_entities[0][0]=="pizza_flavor":
                    self.current_state = "pizza_flavor"
                    return "say_pizza_flavor", org_entities
                if self.current_state=="pizza_sauce" and org_entities[0][0]=="pizza_sauce":
                    return "say_pizza_sauce", org_entities
                if self.current_state=="pizza_toppings" and org_entities[0][0]=="pizza_toppings":
                    self.current_state = "pizza_toppings"
                    return "say_pizza_toppings", org_entities
                if self.current_state==intent_dict[intent]:
                    return intent, org_entities
                if self.current_state=="" and org_entities[0][0]=="pizza_quantity":
                    return "say_pizza_quantity", org_entities
                if self.current_state=="" and "pizza" in intent:
                    return intent, org_entities
                if self.current_state!=intent_dict[intent]:
                    print("**RETURNED HERE**")
                    return "ask_again",[]
                else:
                    return False, []
            else:
                if len(org_entities)==1 and org_entities[0][0]=="menu_item" and org_entities[0][1]=="pizza":
                    return "order_pizza", org_entities
                if self.current_state=="pizza_sauce" and intent=="ask_side_item":
                    return "ask_pizza_sauce", org_entities
                else:
                    return intent, org_entities
        else:
            if self.current_state=="pizza_sauce" and intent=="ask_side_item":
                return "ask_pizza_sauce", org_entities
            elif intent in ["ask_pizza_flavor","ask_pizza_toppings"]:
                if ("flavor" in message or intent=="ask_pizza_flavor") and "topping" not in message:
                    return "ask_pizza_flavor", org_entities
                elif ("topping" in message or intent=="ask_pizza_toppings") and "flavor" not in message:
                    return "ask_pizza_toppings", org_entities
            else:
                return intent, org_entities
        
    
    #Find the entity from the db for the intent
    def find_entity_for_intent(self, message, prev_intent, entities):
        found_entity=[]
        non_pizz_intent_list = [("say_delivery_type", "order_type")]
        say_intent_list = [("say_pizza_size", "pizza_size"), ("say_pizza_sauce", "pizza_sauce"), ("say_sauce", "dipping_sauce"), ("say_pizza_toppings", "pizza_toppings"), ("say_pizza_toppings", "pizza_extra_toppings"), ("say_pizza_crust", "pizza_crust"), ("say_pizza_flavor", "pizza_flavor")]
        ask_intent_list = [("ask_pizza_size", "pizza_size"), ("ask_pizza_sauce", "pizza_sauce"), ("ask_sauce", "dipping_sauce"), ("ask_pizza_toppings", "pizza_toppings"), ("ask_pizza_toppings", "pizza_extra_toppings"), ("ask_pizza_crust", "pizza_crust"), ("ask_pizza_flavor", "pizza_flavor")]

        check_entities = [i for i in entities if 'menu_item' not in i]
        entity_exists = set([i[0] for i in check_entities])
        if len(entity_exists)>1:
            if self.current_state=="pizza_flavor":
                if "pizza_toppings" in entity_exists:
                    prev_intent = "say_pizza_toppings"
                    pass
                else:
                    return "order_pizza", entities
            else:
                return "order_pizza", entities
        
        if prev_intent=="say_pizza_quantity" or self.current_state in ["pizza_quantity"]:
            quantity_list = ["one","two","three","four","five","six"]
            for word in reversed(message.split()):
                if word in quantity_list:
                    return "say_pizza_quantity", [("pizza_quantity",word)]
            return "redirect_to_agent", []
        
        elif any(prev_intent==key for key, value in say_intent_list) or any(prev_intent==key for key, value in ask_intent_list):
            done=False
            if prev_intent in [i[0] for i in say_intent_list]:
                ask_say_intent = "say"
            elif prev_intent in [i[0] for i in ask_intent_list]:
                ask_say_intent = "ask"
                
            message = "wisconsin six cheese" if len(entities)==1 and entities[0][0] == "pizza_flavor" and entities[0][1]=="wisconsin six cheese" else message
            
            if self.current_state=="pizza_flavor" and len(check_entities)>1 and any("pizza_toppings" in i for i in check_entities):
                self.current_state="pizza_toppings"
                prev_intent = "say_pizza_toppings"
            elif self.current_state=="pizza_flavor" and prev_intent=="say_pizza_toppings":
                self.current_state="pizza_toppings"
                prev_intent = "say_pizza_toppings"
            elif self.current_state=="pizza_flavor" and prev_intent=="ask_pizza_toppings":
                self.current_state="pizza_toppings"
                prev_intent = "ask_pizza_toppings"
            
            for n,i in enumerate(range(2)):
                if n==0:
                    if ask_say_intent=="say" and self.current_state != [i[1] for i in say_intent_list if i[0] == prev_intent][0]:
                        intent = [key for key, value in say_intent_list if value == self.current_state][0] if any(value == self.current_state for _, value in say_intent_list) else "else_intent"
                    elif ask_say_intent=="ask" and self.current_state != [i[1] for i in ask_intent_list if i[0] == prev_intent][0]:
                        intent = [key for key, value in ask_intent_list if value == self.current_state][0] if any(value == self.current_state for _, value in say_intent_list) else "else_intent"
                    else:
                        intent = prev_intent
                else:
                    intent = prev_intent
                
                category = intent.split("_")[-1]
                item_list = self.check_items_in_db(menu=True, category=category)
                item_list = item_list.split(",")
                found=""
                for word in message.split():
                    if word not in ["sauce"] and len(word)>=3:
                        found = [item for item in item_list if word in item]
                        if len(found)>1:
                            if any(message in item for item in item_list):
                                found = [next((item for item in item_list if message in item), "")]
                            elif any(self.current_state==key for key, value in entities):
                                found = [next((item for item in item_list if message in item), "")]
                            if found=="" or found==[""]:
                                found = [item for item in item_list if word in item]
                    if word in ["regular"]:
                        found=(word,"")
                    if found!="" and found!=[]:
                        if ask_say_intent=="say":
                            ent = ([value for key, value in say_intent_list if key==intent][0], found[0])
                        elif ask_say_intent=="ask":
                            ent = ([value for key, value in ask_intent_list if key==intent][0], found[0])
                        found_entity.append(ent) if ent not in (found_entity) else found_entity
                        if "toppings" not in intent and "toppings" not in self.current_state:
                            break
                if found_entity!=[]:
                    return intent, found_entity
                else:
                    return intent, []
        
        elif any(prev_intent==key for key, value in non_pizz_intent_list):
            for word in ["take away", "carryout","pickup", "picked up", "pick it up", "carry out"]:
                if word in message:
                    ent = ([value for key, value in non_pizz_intent_list if key==prev_intent][0], word)
                    found_entity.append(ent) if ent not in (found_entity) else found_entity
                    return prev_intent, found_entity
            if found_entity==[]:
                return prev_intent, entities
            else:
                return prev_intent, found_entity
            
        else:
            return prev_intent, entities
        
    def handle_error_response(self, part="first"):
        error_first_part = "Sorry I didn't understand."
        error_second_part = "Please repeat this one more time."
        error_side_item = "Unfortunately, we are not offering this service right now. You can order pizza instead or can talk to our agent."
        error_redirect = "I am forwarding your call to the Headquarter. Kindly let them to assist you."
        
        if part=="first":
            return error_first_part
        elif part=="second":
            return error_second_part
        elif part=="both":
            return error_first_part+error_second_part
        elif part=="side":
            return error_side_item
        elif part=="redirect":
            return error_redirect
        
    def handle_conversational_response(self, condition, item="", category="", no_of_pizza=0):
        greet = "Hi, how may I help you?"
        affirm = "Okay fine."
        ask_order = "What do you want to order?"
        ask_quantity = "How many pizzas would you like to order?"
        ask_for_one_pizza = "Okay. "
        ask_for_more_pizza = f"Okay, Lets start with your {self.pizza_count_in_alpha[no_of_pizza]} pizza. "
        ask_pizza_ingredient = f"What {item} would you prefer?"
        check_pizza_ingredient = f"The {category} you have told is not in our menu.\nPlease select it from these items {item}"
        ask_anything_else = "Do you want anything else with it?"
        ask_delivery_type = "Do you want this order to be carryout or delivery?"
        ask_payment_method = "Do you want to pay with cash or card?"
        ask_order_details = f"Kindly tell me your {item}."
        confirm_order = "Okay your order has been placed. Thanks for calling dominos."
        print_order = f"Ok your order has been confirmed which is {item}."
        tell_pizza_menu = f"We have variety of pizzas. Our most popular pizzas are {item}."+ask_order
        display_items = f"We have {item}. What do you want to order?"
        tell_pizza_ingredient = f"Yes we are offering {item} in our menu. \nDo you want to proceed with this {category}?"
        deny_pizza_ingredient = f"No we are not offering this {category} right now. \nWe have {item}"
        good_bye = "Have a great day. Bye."
        
        if condition=="greet":
            return greet
        elif condition=="affirm":
            return affirm
        elif condition=="ask_order":
            return ask_order
        elif condition=="ask_quantity":
            return ask_quantity
        elif condition=="ask_for_one_pizza":
            return ask_for_one_pizza
        elif condition=="ask_for_more_pizza":
            return ask_for_more_pizza
        elif condition=="ask_pizza_ingredient":
            return ask_pizza_ingredient
        elif condition=="check_pizza_ingredient":
            return check_pizza_ingredient
        elif condition=="ask_anything_else":
            return ask_anything_else
        elif condition=="ask_delivery_type":
            return ask_delivery_type
        elif condition=="ask_payment_method":
            return ask_payment_method
        elif condition=="ask_order_details":
            return ask_order_details
        elif condition=="confirm_order":
            return confirm_order
        elif condition=="print_order":
            return print_order
        elif condition=="tell_pizza_menu":
            return tell_pizza_menu
        elif condition=="display_items":
            return display_items
        elif condition=="tell_pizza_ingredient":
            return tell_pizza_ingredient
        elif condition=="deny_pizza_ingredient":
            return deny_pizza_ingredient
        elif condition=="good_bye":
            return good_bye
        
    

    #Bot responses according to the customer message
    def bot_flow_logic(self, message):
        
        order_detail_list = ["delivery_type","say_phone_number","phone_number","say_address","address","say_pay_with_cash","say_pay_with_card","payment_method","say_card_number","card_number","say_expiration_date","expiration_date","say_security_code","security_code","say_zip_code","zip_code"]
        ask_intents = ["ask_menu", "ask_pizza_crust", "ask_pizza_flavor", "ask_pizza_size", "ask_pizza_sauce", "ask_pizza_toppings", "ask_side_item"]
        
        self.intent, confidence = self.classify_intent(message)
        entities = self.extract_entities(message)
        
        print("intent:",self.intent)
        print("entities:",entities)
        print("confidence:",confidence)
        
        if confidence>=0.4 or self.current_state=="address":
            if self.intent in ["order_pizza","say_pizza_size"]:
                message = message.replace(" a "," one ")
                word_to_num = ["one", "two", "three", "four", "five", "six"]
                message = " ".join([w for i, w in enumerate(message.split()) if w != "one" or (i == 0 or message.split()[i-1] not in word_to_num)])
                entities = self.extract_entities(message)
            self.intent, entities = self.correct_intent_entity(self.intent, confidence, entities, message)
            print("returned intent 1:",self.intent)
            print("returned entities 1:",entities)
            if self.intent==False:
                query = self.check_pizza_entry()
                return self.handle_error_response(part="both")
            elif self.intent=="ask_again":
                return self.handle_error_response(part="first")+self.prev_response
            elif self.intent!=False and ((len(entities)==0 and self.intent in ["say_pizza_size", "say_pizza_sauce", "say_sauce", "say_pizza_toppings", "say_pizza_crust", "say_pizza_quantity", "say_pizza_flavor", "say_side_item"]) or (self.current_state in ["pizza_quantity", "pizza_size", "pizza_sauce", "pizza_toppings", "pizza_crust", "pizza_quantity", "pizza_flavor"])) and self.current_state not in order_detail_list:
                if len(self.selected_item):
                    entities += self.selected_item
                self.intent, entities = self.find_entity_for_intent(message, self.intent, entities)
                print("returned intent 2:",self.intent)
                print("returned entities 2:",entities)
                if len(entities)==0 and self.intent!="redirect_to_agent" and self.intent not in ask_intents:
                    query = self.check_pizza_entry()
                    return self.handle_error_response(part="both")
            elif (self.intent!=False and (len(entities)==0 and self.intent in ["say_delivery_type"])) or (self.current_state in ["delivery_type"]) and self.current_state not in order_detail_list:
                self.intent, entities = self.find_entity_for_intent(message, self.intent, entities)
                print("returned intent 2:",self.intent)
                print("returned entities 2:",entities)
                if len(entities)==0:
                    return self.handle_error_response(part="first")+self.prev_response if len(self.prev_response) else self.handle_error_response(part="both")
        elif confidence<0.4:
            return self.handle_error_response(part="both")
        
        if len(entities):
            self.enter_order_details(entities)
                
        proceed = True
        while proceed:
            proceed = False
            
            intent_query_list = ['ask_pizza_crust','ask_pizza_flavor','ask_pizza_size','ask_pizza_toppings',
                                 'ask_side_item', 'say_pizza_crust','say_pizza_flavor','say_pizza_quantity',
                                 'say_pizza_sauce','say_pizza_size', 'say_pizza_topping','say_sauce',
                                 'say_side_item']
            if self.intent in intent_query_list:
                if self.is_asking(message):
                    self.intent = self.intent.replace('say_','ask_')
                else:
                    self.intent = self.intent.replace('ask_','say_')
            print("returned intent 3:",self.intent)
            
            if self.intent in ["affirm","deny"]:
                if self.intent=="affirm":
                    if self.current_state=="pizza_extra_toppings":
                        return self.handle_conversational_response(condition="ask_quantity", category="topping")
                    elif self.current_state=="side_order":
                        return self.handle_error_response(part="side")
                    elif self.prev_intent in ["ask_pizza_flavor","ask_pizza_crust","ask_pizza_toppings","ask_pizza_sauce","ask_pizza_size"]:
                        entities += self.selected_item
                        if self.begin_pizza_order==True:
                            self.intent = self.prev_intent.replace("ask","say")
                        else:
                            self.intent = "order_pizza"
                        self.selected_item="" if len(self.pizza_list) else self.selected_item
                        proceed=True
                elif self.intent=="deny":
                    if self.current_state=="pizza_extra_toppings":
                        response = self.enter_pizza_info("deny", entities, "pizza_extra_toppings")
                        if response==True and self.comp_pizza_order==True:
                            self.current_state = "side_order"
                            return self.handle_conversational_response(condition="ask_anything_else")
                        else:
                            self.comp_pizza_order = False
                            return response
                    elif self.current_state=="side_order":
                        self.order_completed()
                        self.current_state="delivery_type"
                    elif self.prev_intent in ["ask_pizza_flavor","ask_pizza_crust","ask_pizza_toppings","ask_pizza_sauce","ask_pizza_size"]:
                        self.selected_item=""
                        if len(self.pizza_list):
                            return self.handle_conversational_response(condition="affirm")+self.check_pizza_entry()
            
            elif self.intent=="greet" and self.current_state not in order_detail_list:
                return self.handle_conversational_response(condition="greet")

            elif self.intent=="say_name" and self.current_state not in order_detail_list:
                entities = self.extract_entities(message)
                if any('name' in i for i in entities):
                    self.order_details["name"] = [i[1] for i in entities if i[0] == 'name']
                if any('menu_item' in i for i in entities) or any('pizza_size' in i for i in entities):
                    self.intent="order_pizza"
                    proceed=True
                else:
                    if len(self.pizza_list):
                        pass
                    else:
                        return self.handle_conversational_response(condition="ask_order")

            elif self.intent=="redirect_to_agent":
                self.call_ended=True
                return self.handle_error_response(part="redirect")

            elif self.intent=="order_pizza" and self.current_state not in order_detail_list:
                if self.current_state=="" or len(self.pizza_list)==0:
                    self.current_state = "pizza_quantity"
                    response = self.no_of_pizzas_entry(message, entities)
                    if response!=True and response!="redirect_to_agent":
                        return response
                elif self.current_state!="":
                    response = self.enter_pizza_info(message, entities, self.current_state)
                    if response!=True:
                        return response
                if response==True and self.comp_pizza_order==True:
                    self.current_state = "side_order"
                    return self.handle_conversational_response(condition="ask_anything_else")
                elif response=="redirect_to_agent":
                    proceed=True
                    self.intent="redirect_to_agent"
                else:
                    self.comp_pizza_order = False

            elif self.intent=="say_pizza_quantity" and self.current_state not in order_detail_list:
                response = self.no_of_pizzas_entry(message, entities)
                if response==True and self.comp_pizza_order==True:
                    self.current_state = "side_order"
                    return self.handle_conversational_response(condition="ask_anything_else")
                elif response=="redirect_to_agent":
                    proceed=True
                    self.intent="redirect_to_agent"
                else:
                    self.comp_pizza_order = False
                    if response!=True:
                        return response

            elif self.intent=="say_pizza_crust" and self.current_state not in order_detail_list and self.begin_pizza_order==True:
                response = self.enter_pizza_info(message, entities, "pizza_crust")
                if response==True and self.comp_pizza_order==True:
                    self.current_state = "side_order"
                    return self.handle_conversational_response(condition="ask_anything_else")
                else:
                    self.comp_pizza_order = False
                    if response!=True:
                        return response

            elif self.intent=="say_pizza_sauce" and self.current_state not in order_detail_list and self.begin_pizza_order==True:
                response = self.enter_pizza_info(message, entities, "pizza_sauce")
                if response==True and self.comp_pizza_order==True:
                    self.current_state = "side_order"
                    return self.handle_conversational_response(condition="ask_anything_else")
                else:
                    self.comp_pizza_order = False
                    if response!=True:
                        return response

            elif self.intent=="say_pizza_size" and self.current_state not in order_detail_list and self.begin_pizza_order==True:
                response = self.enter_pizza_info(message, entities, "pizza_size")
                if response==True and self.comp_pizza_order==True:
                    self.current_state = "side_order"
                    return self.handle_conversational_response(condition="ask_anything_else")
                else:
                    self.comp_pizza_order = False
                    if response!=True:
                        return response

            elif self.intent=="say_pizza_flavor" and self.current_state not in order_detail_list:
                response = self.enter_pizza_info(message, entities, "pizza_flavor")
                if response==True and self.comp_pizza_order==True:
                    self.current_state = "side_order"
                    return self.handle_conversational_response(condition="ask_anything_else")
                else:
                    if len(self.pizza_list)==0 and self.prev_intent in ["ask_pizza_flavor","ask_menu"]:
                        self.selected_item = [("pizza_flavor", next((i[1] for i in entities if i[0]=="pizza_flavor"), ""))]
                        response = self.no_of_pizzas_entry(message, entities)
                    self.comp_pizza_order = False
                    if response!=True:
                        return response

            elif self.intent=="say_pizza_toppings" and self.current_state not in order_detail_list and self.begin_pizza_order==True:
                if self.current_state=="pizza_toppings":
                    response = self.enter_pizza_info(message, entities, "pizza_toppings")
                    if response!=True:
                        return response
                elif self.current_state=="pizza_extra_toppings":
                    response = self.enter_pizza_info(message, entities, "pizza_extra_toppings")
                    if response==True and self.comp_pizza_order==True:
                        self.current_state = "side_order"
                        return self.handle_conversational_response(condition="ask_anything_else")
                    else:
                        return response
                else:
                    self.comp_pizza_order = False

            elif self.intent=="say_delivery_type" and self.begin_pizza_order==True:
                entities = self.extract_entities(message)
                if len(entities):
                    self.order_details["delivery_type"] = entities[-1][-1]
                    if entities[-1][-1] in ["delivery","delivered"]:
                        self.current_state = "phone_number"
                        return self.handle_conversational_response(item="phone number", condition="ask_order_details")
                    elif entities[-1][-1] in ["take away", "carryout","pickup", "picked up", "pick it up", "carry out", "pickup"]:
                        self.current_state = "payment_method"
                        return self.handle_conversational_response(condition="ask_payment_method")
                    else:
                        return self.handle_error_response(part="both")
                if not len(self.pizza_list):
                    return self.handle_conversational_response(condition="ask_order")
                
            #Intents From Order Details List
            elif self.intent in order_detail_list or self.current_state in order_detail_list:
                if self.current_state=="phone_number":
                    self.current_state = "address"
                    self.order_details["phone_number"] = message
                    return self.handle_conversational_response(item="delivery address", condition="ask_order_details")
                
                elif self.current_state=="address":
                    self.current_state = "payment_method"
                    self.order_details["address"] = message
                    return self.handle_conversational_response(condition="ask_payment_method")
                
                elif self.current_state=="payment_method":
                    if self.intent=="say_pay_with_cash":
                        self.order_details["payment_method"] = "cash"
                        self.current_state = "order_completed"
                        return self.handle_conversational_response(condition="confirm_order")
                    elif self.intent=="say_pay_with_card":
                        self.order_details["payment_method"] = {"card":{"card_number":"", "expiry_date":"", "cvv_code":"", "zip_code":""}}
                        self.current_state = "card_number"
                        return self.handle_conversational_response(item="phone number", condition="ask_order_details")
                    
                elif self.current_state=="card_number":
                    self.order_details["payment_method"]["card"]["card_number"] = message
                    self.current_state = "expiration_date"
                    return self.handle_conversational_response(item="expiration date", condition="ask_order_details")
                
                elif self.current_state=="expiration_date":
                    self.order_details["payment_method"]["card"]["expiry_date"] = message
                    self.current_state = "security_code"
                    return self.handle_conversational_response(item="card security code", condition="ask_order_details")
                
                elif self.current_state=="security_code":
                    self.order_details["payment_method"]["card"]["cvv_code"] = message
                    self.current_state = "zip_code"
                    return self.handle_conversational_response(item="zip code", condition="ask_order_details")
                
                elif self.current_state=="zip_code":
                    self.order_details["payment_method"]["card"]["zip_code"] = message
                    self.current_state = "order_completed"
                    return self.handle_conversational_response(condition="confirm_order")
            
            #Intents From Ask Ingredients List
            elif self.intent in ask_intents and self.comp_pizza_order==False:
                if self.intent=="ask_menu":
                    items = self.check_items_in_db(menu=True, category="flavor")
                    items = ",".join(items.split(",")[:3])+" and " + items.split(",")[3]
                    return self.handle_conversational_response(item=items, condition="tell_pizza_menu")
                
                elif self.intent!="ask_menu" and len([i for i in entities if 'menu_item' not in i])>1:
                    self.intent="redirect_to_agent"
                    proceed=True
                
                elif self.intent=="ask_pizza_flavor":
                    items = self.check_items_in_db(menu=True, category="flavor")
                    if len(entities) and "pizza_flavor" in [i[0] for i in entities]:
                        ask_flavor = [i[1] for i in entities if i[0]=="pizza_flavor"][0]
                        self.selected_item = [("pizza_flavor",ask_flavor)]
                        if ask_flavor in items:
                            return self.handle_conversational_response(item=ask_flavor, condition="tell_pizza_ingredient", category="flavor")
                        elif ask_flavor not in items:
                            return self.handle_conversational_response(item=items, condition="deny_pizza_ingredient", category="flavor")
                    else:
                        return self.handle_conversational_response(item=items, condition="display_items")
                    
                elif self.intent=="ask_pizza_crust":
                    items = self.check_items_in_db(menu=True, category="crust")
                    if len(entities) and "pizza_crust" in [i[0] for i in entities]:
                        ask_crust = [i[1] for i in entities if i[0]=="pizza_crust"][0]
                        self.selected_item = [("pizza_crust",ask_crust)]
                        if ask_crust in items:
                            return self.handle_conversational_response(item=ask_crust, condition="tell_pizza_ingredient", category="crust")
                        elif ask_crust not in items:
                            return self.handle_conversational_response(item=items, condition="deny_pizza_ingredient", category="crust")
                    else:
                        return self.handle_conversational_response(item=items, condition="display_items")
                    
                elif self.intent=="ask_pizza_toppings":
                    items = self.check_items_in_db(menu=True, category="toppings")
                    if len(entities) and "pizza_toppings" in [i[0] for i in entities]:
                        ask_toppings = [i[1] for i in entities if i[0]=="pizza_toppings"][0]
                        self.selected_item = [("pizza_toppings",ask_toppings)]
                        if ask_toppings in items:
                            return self.handle_conversational_response(item=ask_toppings, condition="tell_pizza_ingredient", category="topping")
                        elif ask_toppings not in items:
                            return self.handle_conversational_response(item=items, condition="deny_pizza_ingredient", category="topping")
                    else:
                        return self.handle_conversational_response(item=items, condition="display_items")
                    
                elif self.intent=="ask_pizza_sauce":
                    items = self.check_items_in_db(menu=True, category="sauce")
                    if len(entities) and "pizza_sauce" in [i[0] for i in entities]:
                        ask_sauce = [i[1] for i in entities if i[0]=="pizza_sauce"][0]
                        self.selected_item = [("pizza_sauce",ask_sauce)]
                        if ask_sauce in items:
                            return self.handle_conversational_response(item=ask_sauce, condition="tell_pizza_ingredient", category="sauce")
                        elif ask_sauce not in items:
                            return self.handle_conversational_response(item=items, condition="deny_pizza_ingredient", category="sauce")
                    else:
                        return self.handle_conversational_response(item=items, condition="display_items")
                    
                elif self.intent=="ask_pizza_size":
                    items = self.check_items_in_db(menu=True, category="size")
                    if len(entities) and "pizza_size" in [i[0] for i in entities]:
                        ask_size = [i[1] for i in entities if i[0]=="pizza_size"][0]
                        self.selected_item = [("pizza_size",ask_size)]
                        if ask_size in items:
                            return self.handle_conversational_response(item=ask_size, condition="tell_pizza_ingredient", category="size")
                        elif ask_size not in items:
                            return self.handle_conversational_response(item=items, condition="deny_pizza_ingredient", category="size")
                    else:
                        return self.handle_conversational_response(item=items, condition="display_items")
                    
                elif self.intent=="ask_side_item":
                    return self.handle_error_response(part="side")
            
            elif self.intent=="say_side_item":
                return self.handle_error_response(part="side")
            
            elif self.intent=="ask_delivery_time":
                response = self.inform_delivery_time()
                self.trigger_query=True
                return response

            elif self.intent=="ask_price":
                proceed=True
                intent="redirect_to_agent"

            elif self.intent=="say_goodbye" and self.current_state not in order_detail_list:
                self.call_ended=True
                return self.handle_conversational_response(condition="good_bye")
            
            #Setting the pizza and side order parameters
            self.set_params()
            
            if proceed!=True:
                if self.begin_pizza_order==False:
                    return self.handle_conversational_response(condition="ask_order")

                elif self.begin_pizza_order==True and self.comp_pizza_order==False and len(self.pizza_list):
                    return self.check_pizza_entry()
                
                elif self.begin_pizza_order==True and self.comp_pizza_order==False and len(self.pizza_list)==0:
                    return self.handle_conversational_response(item=items, condition="ask_quantity")
                
                elif self.begin_pizza_order==True and self.comp_pizza_order==True:
                    if self.current_state=="delivery_type":
                        if self.order_details["delivery_type"]=="":
                            next_query = self.handle_conversational_response(condition="ask_delivery_type")
                        elif self.order_details["delivery_type"]!="":
                            if self.order_details["delivery_type"] in ["take away", "carryout","pickup", "picked up", "pick it up"]:
                                self.current_state="payment_method"
                                next_query = self.handle_conversational_response(condition="ask_payment_method")
                            elif self.order_details["delivery_type"] in ["delivery","delivered"]:
                                self.current_state="phone_number"
                                next_query = self.handle_conversational_response(item="phone number", condition="ask_order_details")
                            else:
                                next_query= self.handle_error_response(part="first")+self.handle_conversational_response(condition="ask_delivery_type")
                        return self.handle_conversational_response(item=self.print_current_order(), condition="print_order")+next_query
                    elif self.current_state=="side_order":
                        return self.handle_conversational_response(condition="ask_anything_else")
                    else:
                        return self.handle_error_response(part="first")+self.prev_response if len(self.prev_response) else self.handle_error_response(part="both")
                    
                else:
                    return self.handle_error_response(part="first")+self.prev_response if self.prev_response!="" else self.handle_error_response(part="both")

                    
    #Convert alphanumeric into digits
    def convert_to_digit(self, message):
        digit_to_word = {"0":"zero","1": "one","2": "two","3": "three","4": "four","5": "five","6": "six","7": "seven","8": "eight","9": "nine"}
        result = []
        message = message.replace("/","")
        if message.isdigit() or not any('A' <= char <= 'Z' or 'a' <= char <= 'z' for char in message):
            words = message
        else:
            words = message.split()
        for word in words:
            if word in list(digit_to_word):
                word = digit_to_word[word]
            else:
                for w in words:
                    if word in list(digit_to_word):
                        word = digit_to_word[word]
            result.append(word)
        return " ".join(result)
    
    #Conversation with Chabot
    def conversation(self):
        
        def remove_unwanted_characters(text):
            pattern = r"[^a-zA-Z0-9\'\- ]"
            return re.sub(pattern, "", text).replace("-"," ")
        
        count = len(os.listdir("/home/ai/personal_workspace/kashif/chatbot/test_conversation"))
        self.conversation_file = f"/home/ai/personal_workspace/kashif/chatbot/test_conversation/{str(count)}.txt"
        
        greet=True
                
        while self.call_ended!=True:
            
            if greet==True:
                print("Hello welcome to dominos pizza. How may I help you?")
                greet=False
            
            message = input("\nNew Message:").lower()
            message = remove_unwanted_characters(message)
            message = self.convert_to_digit(message)
            
            if len(message)!="" and any(char.isalpha() for char in message):
                response = self.bot_flow_logic(message)
                self.prev_intent = self.intent
                
                error_list = ["Unfortunately", "I didn't understand", "Your voice has been broken", "You have provided wrong information", "Please tell me what do you want to order tonight", "Sorry I didn't understand"]
                if any(i for i in error_list if i in response):
                    if self.error_count==3:
                        self.call_ended=True
                        response = self.handle_error_response(part="redirect")
                    else:
                        self.error_count+=1
                else:
                    self.error_count=0
                    prev_intent = response.replace("I didn't understand. Please say that again.","").replace("I didn't understand","").replace("Your voice has been broken.","").replace("You have provided wrong information.","").replace("Sorry I didn't understand.","").replace("Please repeat this one more time","").split("\n")[-1].split(".")[-1].strip()
                    if prev_intent!="":
                        self.prev_response = prev_intent

            else:
                response = self.handle_error_response(part="both")
                
            print(response)
            
            if self.current_state=="order_completed":
                self.call_ended=True

intent_classifier_path = "/home/ai/personal_workspace/kashif/chatbot/models_v0.2/intent_classification/intent_classifier.h5"
intent_labels = "/home/ai/personal_workspace/kashif/chatbot/models_v0.2/intent_classification/labels.pkl"
ner_model_path = "/home/ai/personal_workspace/kashif/chatbot/models/entity_recognition"
dominos_db_path = "/home/ai/personal_workspace/kashif/chatbot/dominos_db.json"

with open(dominos_db_path,"r") as j:
    dominos_db = json.load(j)

bot_obj = clarity_chatbot(intent_classifier_path, intent_labels, ner_model_path, dominos_db)

if __name__ == "__main__":
    bot_obj.conversation()

