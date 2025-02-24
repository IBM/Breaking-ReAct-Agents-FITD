import json
import utils



data = utils.read_all_json_formats('/home/il016930/InjectAgent_to_post_final/fitd_data/test_cases_dh_base.json')
# remove the 'Harmless Thought' field from each dictionary

def mod_1():
    for item in data:
        item.pop('Harmless Thought', None)

    #save the new dict
    with open('/home/il016930/InjectAgent_to_post_final/fitd_data/test_cases_dh_base_no_ht.json', 'w') as file:
        json.dump(data, file, indent=4)


def mod_2():
    # get all the injected_thoughts to different dict and save them
    injected_thoughts = []
    for item in data:
        injected_thoughts.append(item['Injected Thought'])
    
#save the new dict
    with open('/home/il016930/InjectAgent_to_post_final/fitd_data/injected_thoughts.json', 'w') as file:
        json.dump(injected_thoughts, file, indent=4)

mod_2()