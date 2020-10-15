%% load mat file with results 

% specify which depiction to use -
% photo/drawing/sketch_results_mat_corrected.mat

load('sketch_results_mat_corrected.mat')

%% get image labels 

% specify path where stimuli are stored 

photo_path = 'C:\Users\Johannes\Documents\Leipzig\Modelling\Stimuli\ecoset\scaled\photos';
fp = photo_path;
fntmp = dir(fullfile(fp, '*.jpg'));
photo_fn = {fntmp.name}';
photo_filenames = cellfun(@(x) fullfile(fp,x),photo_fn,'uniformoutput',0);
for i = 1:length(photo_filenames)
[~,img_names{i},~] = fileparts(photo_filenames{i});
end 

%% format results array


% remove quotation marks 
results_label = cellfun(@char, results_mat(:,1:15), 'uniformoutput',0);
results_input = results_mat(:,16:30);

%% update labels manually

img_labels= img_names;

% for airplane 

img_labels{2,1} = 'aeroplane';
img_labels{3,1} = 'plane';
img_labels{4,1} = 'jet';
img_labels{5,1} = 'airliner';


% for ant

img_labels{2,3} = 'insect';

% add some synonyms for axe 

img_labels{2,4} = 'hatchet';
img_labels{3,4} = 'ax';
img_labels{4,4} = 'maul';

% for backpack

img_labels{2,5} = 'travel bag';
img_labels{3,5} = 'bag';
img_labels{4,5} = 'pack';
img_labels{5,5} = 'sack';
img_labels{6,5} = 'Bag';

% for bench

img_labels{2,6} = 'bank';

% for bird 

img_labels{2,7} = 'hummingbird';

% for bucket

img_labels{2,9} = 'pail';

% for butterfly

img_labels{2,10} = 'monarch';

%for camel 

img_labels{2,11} = 'dromedary';

% for car
img_labels{2,14} = 'automobile';

% for casette
img_labels{2,15} = 'tape';

% for chicken
img_labels{2,18} = 'rooster';
img_labels{3,18} = 'cock';
img_labels{4,18} = 'hen';


% for computer 

img_labels{2,20} = 'laptop';
img_labels{3,20} = 'notebook';

% for corn

img_labels{2,21} = 'ear of corn';
img_labels{3,21} = 'cob';
img_labels{4,21} = 'maize';

% for couch 

img_labels{2,22} = 'sofa';

% for couch 

img_labels{2,23} = 'lobster';
img_labels{3,23} = 'prawn';

% for donkey

img_labels{2,26} = 'mule';
img_labels{3,26} = 'jack';

% for fish

img_labels{2,28} = 'goldfish';

% for frog

img_labels{2,30} = 'toad';

% for hamburger

img_labels{2,33} = 'cheeseburger';
img_labels{3,33} = 'burger';

% for horse 

img_labels{2,34} = 'sorrel';

% for hourglass

img_labels{2,35} = 'hour glass';
img_labels{3,35} = 'timer';
img_labels{4,35} = 'sand clock';

% for lamp

img_labels{2,36} = 'light';

% for lightbulb

img_labels{2,37} = 'bulb';

% for mailbox

img_labels{2,38} = 'mail box';
img_labels{3,38} = 'letter box';
img_labels{4,38} = 'post box';
img_labels{5,38} = 'postbox';


% for microphone

img_labels{2,39} = 'mike';
img_labels{3,39} = 'mic';

% for pie

img_labels{2,40} = 'cake';

% for pliers

img_labels{2,44} = 'tool';


% for pumpkin

img_labels{2,45} = 'pumkin';

% for rabbit

img_labels{2,46} = 'bunny';

% for rhino

img_labels{2,47} = 'rhinoceros';

% for rifle

img_labels{2,48} = 'gun';
img_labels{3,48} = 'firearm';
img_labels{4,48} = 'weapon';
img_labels{5,48} = 'sniper';

% for saltshaker

img_labels{2,49} = 'salt';
img_labels{3,49} = 'shaker';
img_labels{4,49} = 'salt shaker';
img_labels{5,49} = 'salt';
img_labels{6,49} = 'pepper';
img_labels{7,49} = 'seasoning';

% for scissors

img_labels{2,50} = 'scissor';
img_labels{3,50} = 'shears';

% for seashell

img_labels{2,51} = 'shell';
img_labels{3,51} = 'clam';

% for starfish

img_labels{2,52} = 'star fish';
img_labels{3,52} = 'star';

% for teapot

img_labels{2,54} = 'pot';
img_labels{3,54} = 'kettle';
img_labels{4,54} = 'jug';
img_labels{5,54} = 'jar';

% for telephone

img_labels{2,55} = 'phone';

% for turtle

img_labels{2,57} = 'tortoise';
img_labels{3,57} = 'painted turtle';

% for typewriter

img_labels{2,58} = 'type writer';
img_labels{3,58} = 'typing machine';

%% setup a results table 

% initialize results table 

results_table = table(zeros(60,1), zeros(60,1), cell(60,40));
results_table.Properties.VariableNames = {'Names', 'Accuracy', 'Answer'};
results_table.Names = img_names';

%% calculate results 

for input = 1:15 
    
    for this_HIT = 1:160 
        
        
        correct(this_HIT,input) = sum(strcmpi(img_labels(:,results_input{this_HIT,input}),results_label(this_HIT,input))) | ...
           sum(contains(img_labels(~cellfun('isempty',img_labels(:,results_input{this_HIT,input})),results_input{this_HIT,input}),lower(results_label{this_HIT,input})))|...
           sum(contains(lower(results_label(this_HIT,input)),img_labels(~cellfun('isempty',img_labels(:,results_input{this_HIT,input})),results_input{this_HIT,input})));
       
       if correct(this_HIT, input)
           results_table.Accuracy(results_input{this_HIT,input}) = results_table.Accuracy(results_input{this_HIT,input})+1;
       else 
           results_table.Answer(results_input{this_HIT,input},sum(~cellfun('isempty', results_table.Answer(results_input{this_HIT,input},:)))+1) = results_label(this_HIT,input);
       end
    end 
end 

%% save results 

save('results_table_label_sketches', 'results_table');

