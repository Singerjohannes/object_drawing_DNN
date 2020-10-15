clear all 

% specify for which condition the results should be computed - photos,
% drawings, sketches
cond = 'photos';

% cd to the folder where the results from the MTurk Triplet experiment are
% saved 

switch cond
    case 'photos'
        
       res_paths = {'photos_batch01.csv','photos_batch02.csv','photos_batch03.csv', 'photos_batch04.csv' };

    case 'drawings'
        
       res_paths = {'drawings_batch01.csv','drawings_batch02.csv','drawings_batch03.csv', 'drawings_batch04.csv' };
       
    case 'sketches'
        
       res_paths = {'sketches_batch01.csv','sketches_batch02.csv','sketches_batch03.csv', 'sketches_batch04.csv' };
       
    case 'all'
        
        res_paths = {'photos_batch01.csv','photos_batch02.csv','photos_batch03.csv', 'photos_batch04.csv',...
            'drawings_batch01.csv','drawings_batch02.csv','drawings_batch03.csv', 'drawings_batch04.csv',...
            'sketches_batch01.csv','sketches_batch02.csv','sketches_batch03.csv', 'sketches_batch04.csv'};

end       
       
ALL = {};

for i_path = 1:length(res_paths)
    
    res_path = res_paths{i_path};

h = fopen(res_path);

all = {};
while 1
    l = fgetl(h);
    if l == -1
        break
    end
    all{end+1} = l;
end

fclose(h);

for i = 1:length(all)
    all{i} = strrep(all{i},'""','" "'); % make empty cells not empty -> strsplit messes this up elsewise
    all{i} = strrep(all{i},'","','&&&'); % use &&& as unique identifier
    
    
    all{i}(1) = []; % will remove "
    all{i}(end) = [];
end

all{1} = strrep(all{1},'.','_'); % replace . for field names
names = strsplit(all{1},'&&&');
names(end-1:end) = []; % remove Approve and Reject fields

all(1) = []; % remove headers

ALL = [ALL all];

end

all = ALL;

clear str ALL

for i = 1:length(all)
    
    curr_res = strsplit(all{i},'&&&');
    
    for j = 1:length(names)
        if any(strfind(names{j},'Answer')) % expand those values
            str(i).(names{j}) = strsplit(curr_res{j},' ');
            % for "question", we want to split the links first
        elseif any(strfind(names{j},'WorkerId'))
            str(i).(names{j}) = curr_res{j};
        elseif any(strfind(names{j},'AcceptTime'))
            str(i).(names{j}) = curr_res{j};
        elseif any(strfind(names{j},'SubmitTime'))
            str(i).(names{j}) = curr_res{j};
        elseif any(strfind(names{j},'HITId'))
            str(i).(names{j}) = curr_res{j};    
        else
            
            str(i).Mturk.(names{j}) = curr_res{j};
        
        end
    end
    
    dates = strsplit(str(i).AcceptTime);
    startdate = datevec(sprintf('%02i-%s-%s %s',str2double(dates{3}),dates{2},dates{6},dates{4}));
    dates = strsplit(str(i).SubmitTime);
    enddate = datevec(sprintf('%02i-%s-%s %s',str2double(dates{3}),dates{2},dates{6},dates{4}));
    
    str(i).duration_in_seconds = etime(enddate,startdate);
    
end

% now convert relevant fields
for i = 1:length(str)
    str(i).exp_duration = str2double(str(i).Answer_expDuration)/1000;
    str(i).RT = str2double(str(i).Answer_RT);
    str(i).choice = str2double(strrep(str(i).Answer_question,'link',''));
    str(i).imlink1 = str2double(str(i).Answer_imLink1);
    str(i).imlink2 = str2double(str(i).Answer_imLink2);
    str(i).imlink3 = str2double(str(i).Answer_imLink3);
end


% remove fields 
str = rmfield(str,{'Answer_expDuration','Answer_RT','Answer_imLink1','Answer_imLink2','Answer_imLink3','Answer_question'});

%% Get demographics

% age
for i = 1:length(str)
    tmp = str2num(str(i).Answer_age{1});
    if strcmp(num2str(tmp),str(i).Answer_age)
        age(i,1) = tmp;
    else
        age(i,1) = NaN;
    end
end

% gender
for i = 1:length(str)
    if strcmp(str(i).Answer_gender,'male')
        gender(i,1) = 0;
    elseif strcmp(str(i).Answer_gender,'female')
        gender(i,1) = 1;
    elseif strcmp(str(i).Answer_gender,'other')
        gender(i,1) = 2;
    else
        gender(i,1) = NaN;
    end
end

% workerID
for i = 1:length(str)
    id{i,1} = str(i).WorkerId;
end

% Now get number of participants, their age and gender, mean number of HITs per
% person, median duration and therefore pay per hour
[uid,~,uid_ind] = unique(id);
n_worker = length(uid);
h = hist(uid_ind,1:n_worker);

mean_hit_per_worker = mean(h);

for i_worker = 1:n_worker
    workerind = find(strcmp(id,uid{i_worker}));
age_dist(i_worker,1) = age(workerind(1));
gender_dist(i_worker,1) = gender(workerind(1));
end

mean_age = mean(age_dist(age_dist<100));
max_age = max(age_dist(age_dist<100));
min_age = min(age_dist(age_dist<100));

n_female = sum(gender_dist==1);
n_male = sum(gender_dist==0);

% cap RTs to 12s because some are unrealistic
%RTs_capped = min(RTs,12000)/1000; % doesn't make a difference really

fprintf('Number of participants: %i\n',n_worker)
fprintf('Mean number of trials per worker: %.2f (min: %i, max: %i)\n',mean_hit_per_worker*20,min(h)*20,max(h)*20)
%fprintf('Payment per hour based on median RT of %.2f: USD%.2f\n',median(RTs_capped(:)),0.005 * 3600/median(RTs_capped(:)))
fprintf('Age: %.2f (range: %i - %i, std: %.2f)\n',mean_age, min_age, max_age, std(age_dist))
fprintf('Gender: %i female, %i male\n',n_female,n_male)
%fprintf('Mean RT over all HITs was %f\n', mean(mean(RTs_capped)));


%% extract response values into matrix
mat = zeros(60,60);
cnt = eye(60,60); % eye will help prevent NaN and will lead to 0 diagonal
lst = zeros(length(str)*20*6,4);
ct = -5;

currcnt = 0;

for i = 1:length(str)
    
   
    for j = 1:length(str(i).imlink1)
        
        triplet = [str(i).imlink1(j) str(i).imlink2(j) str(i).imlink3(j)];
        currcnt = currcnt+1;

        
        if triplet(1)==triplet(2)||triplet(1)==triplet(3)||triplet(2)==triplet(3)
            continue
        end
        
        if length(str(i).choice)<20
            continue
        end
        choice = str(i).choice(j);
        
        triplets(currcnt,:) = [triplet choice RTs(i,j)];

        
        if isnan(choice)
            continue
        end
        
        if any(triplet>60)
           continue
        end
        
        ct = ct+6;
        
        ind = setdiff(1:3,choice);
        % update the similarity matrix
        mat(triplet(ind(1)),triplet(ind(2))) = mat(triplet(ind(1)),triplet(ind(2))) + 1;
        mat(triplet(ind(2)),triplet(ind(1))) = mat(triplet(ind(2)),triplet(ind(1))) + 1;
        % update the number of occurrence matrix
        cnt(triplet(1),triplet(2)) = cnt(triplet(1),triplet(2)) + 1;
        cnt(triplet(1),triplet(3)) = cnt(triplet(1),triplet(3)) + 1;
        cnt(triplet(2),triplet(3)) = cnt(triplet(2),triplet(3)) + 1;
        cnt(triplet(2),triplet(1)) = cnt(triplet(2),triplet(1)) + 1;
        cnt(triplet(3),triplet(1)) = cnt(triplet(3),triplet(1)) + 1;
        cnt(triplet(3),triplet(2)) = cnt(triplet(3),triplet(2)) + 1;
        
        % we always need to sample triplets from the same trial
        lst(ct,:) = [triplet(ind(1)) triplet(ind(2)) 1 i];
        lst(ct+1,:) = [triplet(ind(1)) triplet(choice) 0 i];
        lst(ct+2,:) = [triplet(ind(2)) triplet(choice) 0 i];
        lst(ct+3,:) = [triplet(ind(2)) triplet(ind(1)) 1 i];
        lst(ct+4,:) = [triplet(choice) triplet(ind(1)) 0 i];
        lst(ct+5,:) = [triplet(choice) triplet(ind(2)) 0 i];        
    end
    
end

try %#ok<TRYNC>
    lst(ct+6:end,:) = []; % remove trailing lines
end

final_mat = mat./cnt;
% now check if there are NaN
if any(isnan(final_mat(:)))
    warning('matrix contains NaN')
end
% now check if there are still asymmetries
if ~isequal(tril(final_mat),tril(final_mat'))
    warning('matrix not symmetrical, check!')
end

save([cond ,'_mat.mat'],'final_mat','triplets')