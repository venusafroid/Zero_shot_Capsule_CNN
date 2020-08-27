from __future__ import print_function
""" input data preprocess.
"""

import tensorflow
import numpy as np
import tool

from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


data_prefix = '../data/nlu_data/'
word2vec_path = data_prefix+'glove.6B.300d.txt'
# word2vec_path = data_prefix+'wiki.en.vec'
# training_data_path = data_prefix + 'train_shuffle.txt'
# test_data_path = data_prefix + 'test.txt'

training_data_path = '/dfsdata2/hugy3_data/lena_moli/lena_train_shuffle_ht_supintent.txt'
test_data_path = '/dfsdata2/hugy3_data/lena_moli/moli_train_shuffle_ht_supintent.txt'

seen_intent = ['expand additional ram', 'display setting', 'system upgrade/downgrade', 'recovery media', 'windows installation', 'windows activation', 'driver install/uninstall/update', 'system restore/system recovery/system reinstall/factory reset/', 'windows safe mode', 'how to change/expand sim card', 'check for serial number', 'fn hotkeys shortcut description', 'lenovo patch utility', 'bios settings', 'office software', 'hard disk storage full and how to clean up', 'windows system account password', 'how to get bit locker recovery key', 'create system backup', 'bios upgrade', 'apps in windows app store', 'check one key recovery button position', 'internet explorer (ie) browser', 'antivirus software', 'check current windows version', 'lenovo settings', 'system time and date setting', 'how to enter bios', 'expand additional hard drive', 'ubuntu system installation', 'disable sleep mode', 'adobe reader', 'lenovo hardware diagnose tools', 'how to create/delete partitions in disc management', 'system windows defender', 'voice assistant cortana', 'browser version check', 'lenovo cloud', 'how to enable/disable bitlocker', 'enable/disable windows firewall', 'create/delete/edit windows system account', 'multiple operation systems installation', 'create shortcut', 'trusted platform module (tpm)', 'whether the computer can install a specific operating system', 'hard drive optimization tool', 'veriface', 'lenovo companion', 'lenovo solution center', 'how to setup/configure raid', 'how to clean your device', 'update your computer using lenovo vantage', 'parts removal and replacement', 'cleaning the cover of your computer', 'cleaning your computer keyboard', 'cleaning your computer lcd display', 'disinfecting your computer, keyboard and lcd display']
unseen_intent = ['Hot spots and tethering', 'Control app permissions', 'Delete photos or videos', 'Share or receive with NFC', 'Unlock bootloader', 'External reset', 'Back up apps and settings', 'Check for phone software updates', 'Get apps', 'System flash', 'International carrier support', 'Use song file as ringtone', 'Unlock the phone to use with another carrier', 'Recover recently deleted files', 'Phone storage option', 'Use call recording', 'Find IMEI for the phone', 'Forget screen lock, pattern, pin, or password', 'Use voice typing', 'Analyze battery use', 'Factory Data Reset', 'Sync accounts', 'If you lost your phone', 'Manage notifications', 'Insert and remove SIM and SD cards', 'Transfer files between phone and computer', 'Transfer files from previous phone to current phone', 'Stop Pop-up ads', 'Take a photo', 'Send new Email', 'Set ringtone for contact', 'Change intensity, vibrations features', 'Project your screen to TV', 'Turn on call waiting', 'Take and share screenshots', 'Turn on/off keyboard vibrations and sounds', 'Clear cache partition', 'Ask for manual', 'Make video calls', 'Diagnose in safe mode', 'Get USB drivers for PC or Mac', 'Change storage type', 'Set up voicemail', 'Scan documents with google drive', 'Block/unblock calls and texts from contact', 'Move media from internal storage to SD card', 'Change font and display size', 'Find photos and videos', 'Send new message', 'Delete apps', 'show caller id for incoming calls', 'Moto Voice', 'Ask for the previous Google account', 'View or free up phone storage', 'Change the default ringtone', 'Set volte', 'Make calls over Wi-Fi', 'Add Email attachment', 'Register my product', 'Make conference calls', 'Add or remove accounts', 'Record a video', 'If you have visual impairments', 'Pair with a bluetooth device', 'Share contacts', 'Find downloaded files', 'Listen to music', 'Connect to a Wi-Fi network', 'Set system notification of update', 'Set virtual navigation keys', 'Use Battery Saver mode', 'Export and import contacts', 'Use onscreen keyboard', 'Add widgets on home screen', 'Charge phone', 'Relock bootloader', 'Set up Email', 'Manage bluetooth devices list', 'Delete browsing data', 'Manage voicemail mailbox', 'Downgrade system', 'Select SIM for cellular data', 'Connect with a USB OTG cable', 'Manage frequently used words', 'Moto Actions', 'Transfer music files', 'Check Android version', 'Add contacts', 'Find Serial number for the phone', 'Adjust camera settings', 'Set alarms', 'add time stamp to photos', 'Edit photos', 'Delete Emails', 'Manage fingerprints', 'Control which calendar events are shown', 'Restrict background data', 'View and manage Email attachments', 'Use location services and GPS', 'Change phone name for bluetooth', 'Update apps in play store', 'Delete contacts', 'Set notification light', 'Use or stop group messages', 'Force stop apps', 'Add messaging signature', 'Manage screen lock', 'Show battery percentage in status bar', 'Set up SIMs by usage', 'Turn on call roaming', 'Control Email notifications', 'Save attachments from messages', 'Share photos and videos', 'Test for virus/malware', 'Store photos and videos on SD card', 'Use a flash when taking photos', 'Set up emergency information', 'Answer calls', 'Remove widgets, shortcuts, or folders', 'Extend battery life', 'Get phone model', 'Delete or archive messages', 'Use multiple languages', 'Print files from your phone', 'Adjust volume', 'Avoid interruptions with Do not disturb', 'Use Notification dots', 'Choose color mode', 'Use call forwarding', 'Using moto mods', 'Change default usb setting', 'Restrict calls and messages for users', 'Use two apps at once', 'Manage calls', 'Non changeable setting', 'Set up your Motorola ID', 'Add shortcuts on home screen', 'Erase SD card', 'Select SIM for text messages', 'Set network type', 'Set quick responses for calls', 'Control text messages notifications', 'Unlock your screen automatically', 'Help app', 'Clear cache or data of apps', 'Change display of contacts', 'Find Emails', 'Remake your home screen with launchers', 'Listen to FM radio', 'If you have hearing impairments', 'Limit messages in conversation', 'Listen to voicemail', 'Adjust photo size', 'Turn phone off/on', 'Modify a user', 'Optimize screen brightness for available light', 'Time zone', 'Turn off or on emergency alerts', 'Lock SIM card', 'Edit contacts', 'Gmail label', 'Mobile plan, insurance, or billing questions', 'Turn off bluetooth', 'Restore apps from Play Store', 'Turn on/off cellular data', 'hide or show files', 'Copy text from a message', 'Edit videos', 'Use beauty filter', 'Change wallpaper', 'Change Moto Voice privacy permissions', 'Read and reply to a message', 'Use timer or stopwatch', 'Set demo mode', 'Set ascending ringtone', 'Use Airplane mode', 'Preview notifications when screen sleeps', 'Turn Wi-Fi on/off', 'Edit or delete calendar events', 'Add a user', 'Use a phone projector', 'Auto rotate', 'Customize your favorites tray', 'Remove a user', 'View and delete call history', 'Adjust ISO and other parameters', 'Motorola Account Deletion', 'Eject SD card', 'Adjust screen colors at night', 'Use Google Assistant', 'Reconnect with a bluetooth device', 'Guest session', 'Encrypt phone or SD card', 'Enable video stabilization', 'Star your favorite contacts', 'Change format of times or dates', 'Create calendar events', 'Adjust usage settings for paired bluetooth device', 'Remove and replace back cover', 'Set separate volume levels for notifications and ringtones', 'Forward a message', 'Search phone and web', 'Manage lock screen notifications', 'Set date and time', 'Delete internet history', 'Disable Google play service', 'Use Google Maps', 'Adjust system sounds', 'Upload photos and videos', 'Cut, copy, paste text', 'Close recent apps', 'Resize widgets on home screen', 'Tap and pay with your phone', 'Change screen timeout', 'Reset network settings', 'show caller id for outbound calls', 'Adjust screen sensitivity', 'Set cellular data limit', 'Stop moto voice commands', 'Android One', 'Organize photos and videos', 'Connect to VPNs', 'Add folders on home screen', 'Use fingerprint sensor to navigate', 'Record location tag of photo', 'Add attachments in text messages', 'Download photos', 'Scan codes with camera', 'Share from google photos', 'Take black & white photos', 'Use the time and weather widget', 'Merge contacts', 'Add or delete members from group message', 'Take panoramic photos', 'teamviewer support', 'Take a selfie', 'Set auto answer', 'Set up quick responses for Email', 'switch camera from front to rear', 'Use photo as wallpaper or contact photo', 'set mirror image', 'Change default messaging app', 'Use recording as ringtone', 'Use plug-in headset', 'Change default browser', 'Make calls', 'Automatic SIM selection', 'Use macro mode', 'Visit a web site', 'Find calendar events', 'Control Moto Voice notifications', 'use call and cellular data together', 'Use a screen saver', 'Get repair assistant', 'Download ringtones', 'Use night mode when taking photos', 'Set default camera app', 'Read Email', 'Select SIM for calls', 'Prioritize voice or data usage for dual sims', 'Video playback', 'Add Data Saver tile to quick settings', 'Change SIM names and colors for dual SIMs', 'Turn off wireless charging sounds', 'Switch users to share your phone', 'Contact Google', 'Reboot', 'Set up find my device', 'Request delivery reports', 'Playing wav files', 'View notifications', 'Pin your screen', 'Add Email signature', 'Preview inboxes from home screen', 'Reply to or forward an Email', 'Manage text messages drafts', 'Upload music to the cloud', 'Switch browser tabs', 'Use a grid for framing', 'Use a timer when taking photos', 'Use HDR in high-contrast lighting', 'Create highlight reels', 'View in filmstrip mode', 'Open the camera', 'Take depth enabled photos', 'Remove and replace shell', 'Use voice commands for searching', 'Shrink screen for one-handed use', 'Add message to lock screen', 'Get turn-by-turn directions', 'Send calls to voicemail', 'Find my phone number', 'Change device theme', 'Digital Wellbeing', 'Install and setup LMSA', 'Manage LMSA account', 'Use 3rd party apps']

# seen_intent = ['music', 'search', 'movie', 'weather', 'restaurant']
# unseen_intent = ['playlist', 'book']

# unseen_intent = ['connectivity', 'audio and sound']
# unseen_intent = ['Operating System', 'Port']

# seen_intent = ['Troubleshooting_Technical_Issues','How_to_configuration']
# unseen_intent = ['playlist', 'book']
# unseen_intent = ['Trouble Shooting', 'How to']
# seen_intent = ['playlist', 'book', 'movie', 'weather', 'restaurant']
# seen_intent = ['music', 'search']
# unseen_intent = ['music', 'search']

# seen_intent = ['playlist', 'book', 'movie', 'music', 'search']
# unseen_intent = ['weather', 'restaurant']

# seen_intent = ['playlist', 'book', 'weather', 'restaurant', 'search']
# unseen_intent = ['movie', 'music']

# seen_intent = ['music', 'search', 'movie', 'book', 'restaurant']
# unseen_intent = ['playlist', 'weather']

# seen_intent = ['playlist', 'book', 'movie', 'weather', 'search']
# unseen_intent = ['music', 'restaurant']

# seen_intent = ['playlist', 'book', 'restaurant', 'music', 'search']
# unseen_intent = ['weather', 'movie']

# seen_intent = ['playlist', 'weather', 'restaurant', 'music', 'movie']
# unseen_intent = ['book', 'search']

# seen_intent = ['book', 'weather', 'restaurant', 'music', 'movie']
# unseen_intent = ['playlist', 'search']

# seen_intent = ['playlist', 'weather', 'movie', 'search']
# unseen_intent = ['book', 'restaurant', 'music']

# seen_intent = ['book', 'restaurant', 'music', 'search']
# unseen_intent = ['playlist', 'weather', 'movie']

# seen_intent = ['book', 'restaurant', 'weather', 'search']
# unseen_intent = ['playlist', 'music', 'movie']

def load_w2v(file_name):
    """ load w2v model
        input: model file name
        output: w2v model
    """
#     glove2word2vec(file_name, 'glove.6B.300d.txt.tmp')
    w2v = KeyedVectors.load_word2vec_format(
            'glove.6B.300d.txt.tmp', binary=False, unicode_errors='ignore')
#     w2v = KeyedVectors.load_word2vec_format(
#             file_name, binary=False, unicode_errors='ignore')
    return w2v

def process_label(intents, w2v):
    """ pre process class labels
        input: class label file name, w2v model
        output: class dict and label vectors
    """
    class_dict = {}
    label_vec = []
    class_id = 0
    for line in intents:
        # check whether all the words in w2v dict
        if line == 'Troubleshooting_Technical_Issues':
            line = 'Trouble_shooting_Technical_Issues'
        if '_' in line:
            label = [w.lower().strip() for w in line.split('_')]
        else:
            label = [w.lower().strip() for w in line.split(' ')]
        label_new = []
        for w in label:
            if not w.lower() in w2v.vocab:
                label_new += list(w.lower())
            else:
                label_new.append(w.lower())
        label = label_new
        for w in label:
            if not w2v.vocab.has_key(w.lower()):
                print("not in w2v dict", w)
        # compute label vec
        label_sum = np.sum([w2v[w] for w in label], axis = 0)
        label_vec.append(label_sum)
        # store class names => index
        class_dict[' '.join(label)] = class_id
        class_id = class_id + 1
    return class_dict, np.asarray(label_vec)

def load_vec(file_path, w2v, class_dict, in_max_len):
    """ load input data
        input:
            file_path: input data file
            w2v: word2vec model
            max_len: max length of sentence
        output:
            input_x: input sentence word ids
            input_y: input label ids
            s_len: input sentence length
            max_len: max length of sentence
    """
    input_x = [] # input sentence word ids
    input_y = [] # input label ids

    s_len = [] # input sentence length
    max_len = 0

    for line in open(file_path):
        arr = line.strip().split('\t')
        if len(arr) < 2:
            continue
        if '_' in arr[0]:
            label = [w.lower().strip() for w in arr[0].split('_')]
        else:
            label = [w.lower().strip() for w in arr[0].split(' ')]
         
        question = [w for w in arr[1].split(' ')]
        cname = ' '.join(label) 
        if not class_dict.has_key(cname):
            continue

        # trans words into indexes
        x_arr = []
        for w in question:
            if w2v.vocab.has_key(w):
                x_arr.append(w2v.vocab[w].index)
        s_l = len(x_arr)
        if s_l <= 1:
            continue
        if in_max_len == 0:
            if s_l > max_len:
                max_len = len(x_arr)

        input_x.append(np.asarray(x_arr))
        input_y.append(class_dict[cname])  
        s_len.append(s_l)

    # add paddings
    max_len = max(in_max_len, max_len)
    x_padding = []
    for i in range(len(input_x)):
        if (max_len < s_len[i]):
            x_padding.append(input_x[i][0:max_len])
            continue
        tmp = np.append(input_x[i], np.zeros((max_len - s_len[i],), dtype=np.int64))
        x_padding.append(tmp)
    x_padding = np.asarray(x_padding)
    input_y = np.asarray(input_y)
    s_len = np.asarray(s_len)
    return x_padding, input_y, s_len, max_len

def get_label(data):
    Ybase = data['y_tr']
    sample_num = Ybase.shape[0]
    labels = np.unique(Ybase)
    class_num = labels.shape[0]
    labels = range(class_num)
    # get label index
    ind = np.zeros((sample_num, class_num), dtype=np.float32)
    for i in range(class_num):
        ind[Ybase == labels[i], i] = 1;
    return ind

def read_datasets():
    print("------------------read datasets begin-------------------")
    data = {}

    # load word2vec model
    print("------------------load word2vec begin-------------------")
    w2v = load_w2v(word2vec_path)
    print("------------------load word2vec end---------------------")

    # load normalized word embeddings
    embedding = w2v.syn0
    data['embedding'] = embedding
    norm_embedding = tool.norm_matrix(embedding)
    data['embedding'] = norm_embedding
    # pre process seen and unseen labels
    sc_dict, sc_vec = process_label(seen_intent, w2v)
    print("sc_dict", sc_dict)
    uc_dict, uc_vec = process_label(unseen_intent, w2v)

    # trans data into embedding vectors
    max_len = 0
    x_tr, y_tr, s_len, max_len = load_vec(
            training_data_path, w2v, sc_dict, max_len)
    x_te, y_te, u_len, max_len = load_vec(
            test_data_path, w2v, uc_dict, max_len)
    data['x_tr'] = x_tr
    data['y_tr'] = y_tr

    data['s_len'] = s_len
    data['sc_vec'] = sc_vec
    data['sc_dict'] = sc_dict

    data['x_te'] = x_te
    data['y_te'] = y_te

    data['u_len'] = u_len
    data['uc_vec'] = uc_vec
    data['uc_dict'] = uc_dict

    data['max_len'] = max_len

    ind = get_label(data)
    data['s_label'] = ind # [0.0, 0.0, ..., 1.0, ..., 0.0]
    print("------------------read datasets end---------------------")
    return data
