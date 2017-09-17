function [data, features, testData, testFeatures] = getVideo(input, dataset_info, phase)

	path = [input.PATH_DATA dataset_info.path];
    if(strcmpi(phase,'source')) % Kinetics
        path = [path lower(dataset_info.source)];
    elseif(strcmpi(phase,'target')) % UCF
        path = [path lower(dataset_info.target)];
    end
    
    fts = readNPY([path '.npy']);
    ids = readNPY([path '_labels.npy']) + 1;
    % Transform ids to labels
    [new_labels, labels_ids] = castLabels(phase, ids);
    input.([phase 'Dataset']).classes = sort_nat(new_labels);
    data.annotations.classes = labels_ids;

    % NOTE: Change everytime we run new experiments to get new results
    if(input.seedRand > 0) % For different semi-supervised sets
        rng(input.seedRand);
    end

    if(input.isZScore)
        fts = fts./ repmat(sum(fts,2),1,size(fts,2));
        fts = zscore(fts);
    end
    features = fts;
    
    % Less samples
%     if(strcmpi(phase,'target'))
%         numSamples = 5000;
%         randSamples  = randperm(size(features,1));
%         data.annotations.classes = data.annotations.classes(randSamples(1:numSamples));
%         features = features(randSamples(1:numSamples),:);
%     end
    if(strcmpi(phase,'target'))
        permSamples = randperm(length(labels_ids));
        data.annotations.classes = data.annotations.classes(permSamples);
        features = features(permSamples,:);
    else % source
%         permSamples = randperm(length(ids));
%         features = features(permSamples(1:10000),:);
%         data.annotations.classes = data.annotations.classes(permSamples(1:10000));
    end
    
    testFeatures = [];
    testData.annotations.classes = [];
    if(isempty(testData.annotations.classes))
        testData = [];
    end
    
end

function [new_labels, port_ids] = castLabels(phase, ids)

    src_labels = {'abseiling' 'air drumming' 'answering question' 'applauding' 'applying cream' 'archery' 'arm wrestling' ...
        'arranging flowers' 'assembling computer' 'auctioning' 'baby walking up' 'baking cookies' 'balloon blowing' ...
        'bandaging' 'barbequing' 'bartending' 'beatboxing'  'bee keeping' 'belly dancing' 'bench pressing' 'bending back' ...
        'bending metal' 'biking through snow' 'blasting sand' 'blowing glass'  'blowing leaves' 'blowing nose' ...
        'blowing out candles' 'bobsledding' 'bookbinding' 'bouncing on trampolin' 'bowling' 'braiding hair' .... 
        'breading or breadcrumbing' 'breakdancing' 'brush painting' 'brushing hair' 'brushing teeth' 'building cabinet' ...
        'building shed' 'bungee jumping' 'busking' 'canoeing or kayaking' 'capoeira' 'carrying baby' 'cartwheeling' ...
        'carving pumpkin' 'catching fish' 'catching or throwing baseball' 'catching or throwing frisbee' 'catching or throwing softball' ...
        'calebrating' 'changing oil' 'changing wheel' 'checking tires' 'cheerleading' 'chopping wood'  'clapping' 'clay pottery making' ...
        'clean and jerk' 'cleaning floor' 'cleaning gutters' 'cleaning pool' 'cleaning shoes' 'cleaning toilet' ...
        'cleaning windows' 'climbing a rope' 'climbing ladder' 'climbing tree' 'contact juggling' 'cooking chicken' ...
        'cooking egg' 'cooking on campfire' 'cooking sausages' 'counting money' 'contry line dancing' 'cracking neck' ...
        'crawling baby' 'crossing river' 'crying' 'curling hair' 'cutting nails'  'cutting pineapple' 'cutting watermelon' ...
        'dancing ballet' 'dancing charleston' 'dancing gangnam' 'dancing macarena' 'deadlifting' 'decorating the christmas tree' 'digging' ...
        'dining' 'disc golfing' 'diving cliff' 'dodgeball' 'doing aerobics' 'doing laundry' 'doing nails' 'drawing' ...
        'dribbling basketball' 'drinking' 'drinking beer' 'drinking shots' 'driving car' 'driving tractor' 'drop kicking' ...
        'drumming fingers' 'dunking basketball' 'dying hair' 'eating burger' 'eating cake' 'eating carrots' 'eating chips' ...
        'eating doughnuts' 'eating hotdog' 'eating ice cream' 'eating spaghetti' 'eating watermelon' 'egg hunting' 'exercising arm' ...
        'exercising with an exercise ball' 'extinguishing fire' 'faceplanting' 'feeding birds' 'feeding fish' 'feeding goats' ...
        'filling eyebrows' 'finger snapping' 'fixing hair' 'flipping pancake' 'flying kite' 'folding clothes' 'folding napkins' ...
        'folding paper' 'front raises' 'frying vegetables' 'garbage collecting' 'gargling' 'getting a haircut'  'getting a tattoo' ...
        'giving or receiving award' 'golf chipping' 'golf driving' 'golf putting' 'grinding meat' 'grooming dog' 'grooming horse' ...
        'gymnastics tumbling' 'hammer throw' 'headbanging' 'headbutting'  'high jump' 'high kick' 'hitting baseball' ...
        'hockey stop' 'holding snake' 'hopscotch' 'hoverboarding' 'hugging' 'hula hooping' 'hurdling' 'hurling (sport)' 'ice climbing' ...
        'ice fishing' 'ice skating' 'ironing' 'javelin throw' 'jetskiing' 'jogging' 'juggling balls' 'juggling fire' ...
        'juggling soccer ball' 'jumping into pool' 'jumpstyle dancing' 'kicking field goal' 'kicking soccer ball' 'kissing' ...
        'kitesurfing' 'knitting' 'krumping' 'laughing' 'laying bricks' 'long jump' 'lunge' 'making a cake' 'making a sandwich' ...
        'making bed' 'making jewerly' 'making pizza' 'making snowman' 'making sushi' 'making tea' 'marching' 'massaging back' ...
        'massaging feet' 'massaging legs' 'massaging persons head' 'milking cow'  'mopping floor' 'motorcycling' 'moving furniture' ...
        'mowing lawn' 'news anchoring' 'opening bottle' 'opening present' 'paragliding' 'parasailing' 'parkour' 'passing American football (in game)' ...
        'passing American football (not in game)' 'peeling apples' 'peeling potatoes' 'petting animal (not cat)' 'petting cat' ...
        'picking fruit' 'planting trees' 'plastering' 'playing accordion' 'playing badminton' 'playing bagpipes' 'playing basketball' ...
        'playing bass guitar' 'playing cards' 'playing cello' 'playing chess' 'playing clarinet' 'playing controller' 'playing cricket' ...
        'playing cymbals' 'playing didgeridoo' 'playing drums' 'playing flute' 'playing guitar' 'playing harmonica' 'playing harp' ...
        'playing ice hockey' 'playing keyboard' 'playing kickball' 'playing monopoly' 'playing organ' 'playing paintball' 'playing piano' ...
        'playing poker' 'playing recorder' 'playing saxophone' 'playing squash or racquetball' 'playing tennis' 'playing trombone' 'playing trumpet' ...
        'playing ukulele' 'playing violin' 'playing volleyball' 'playing xylophone' 'pole vault' 'presenting weather forecast' ...
        'pull ups' 'pumping fist' 'pumping gas' 'punching bag' 'punching person (boxing)' 'push up' 'pushing car' 'pushign cart' ...
        'pushing wheelchair' 'reading book' 'reading newspaper' 'recording music' 'riding a bike' 'ridin a camel' 'riding elephant' 'riding mechanical bull' ...
        'riding mountain bike' 'riding mule' 'riding or walking with horse' 'riding scooter' 'riding unicycle' 'ripping paper' 'robot dancing' ...
        'rock climbing' 'rock scissors paper' 'roller skating' 'running on treadmill' 'sailing' 'salsa dancing' 'sanding floor' ...
        'scrambling eggs' 'scuba diving' 'setting table' 'shaking hands' 'shaking head' 'sharpening knives' 'sharpening pencil' 'shaving head' ...
        'shaving legs' 'shearing sheep' 'shining shoes' 'shooting basketball' 'shooting goal (soccer)' 'shot put' 'shoveling snow' ...
        'shredding paper' 'shuffling cards' 'side kick' 'sign language interpreting' 'singing' 'situp' 'skateboarding' 'ski jumping' ...
        'skiing (not slalom or crosscountry)' 'skiing crosscountry' 'skiing slalom' 'skipping rope' 'skydiving' 'slacklining' 'slapping' ...
        'sled dog racing' 'smoking' 'smoking hookah' 'snatch weight lifting' 'sneezing' 'sniffing' 'snorkeling' 'snowboarding' ...
        'snowkiting' 'snowmobiling' 'smoersaulting' 'spinning poi' 'spray painting' 'spraying' 'springboard diving' 'squat' ...
        'stickin tongue' 'stomping grapes'  'stretching arm' 'stretching leg' 'strumming guitar' 'surfing crowd' 'surfing water' 'sweeping floor' ...
        'swimming backstroke' 'swimming breast stroke' 'swimming butterfly stroke' 'swing dancing' 'swinging legs' 'swinging on something' ...
        'sword fighting' 'tai chi' 'taking a shower' 'tango dancing' 'tap dancing' 'tapping guitar' 'tapping pen' 'tasting beer' ...
        'tasting food' 'testfying' 'texting' 'throwing axe' 'throwing ball' 'throwing discus' 'tickling' 'tobogganing' 'tossing coin' ...
        'tossing salad' 'training dog' 'trapezing' 'trimming or shaving beard' 'trimming trees' 'triple jump' 'tying bow tie' ...
        'tying know (not on a tie)' 'tying tie' 'unboxing' 'unloading truck' 'using computer' 'using remote controller (not gaming)' 'using segway' ...
        'vault' 'waiting in line' 'walking the dog' 'washing dishes' 'washing feet' 'washing hair'  'washing hands' 'water skiing' ...
        'water sliding' 'watering plants' 'waxing back' 'waxing chest' 'waxing eyebrows' 'waxing legs' 'weaving basket' ...
        'welding' 'whistling' 'windsurfing' 'wrapping present' 'wrestling' 'writing' 'yawning' 'yoga' 'zumba'};
    tgt_labels = {'apply eye makeup' 'apply lipstick' 'blow dry hair' 'brushing teeth' 'cutting in kitchen' 'hammering' 'hula hoop' ...
        'juggling balls' 'jump rope' 'knitting' 'mixing batter' 'mopping floor' 'nun chucks' 'pizza tossing' 'shaving beard' ...
        'skate boarding' 'soccer juggling' 'typing' 'writing on board' 'yo yo' 'baby crawling' 'blowing candles' ...
        'body weight squats' 'handstand pushups' 'handstand walking' 'jumping jack' 'lunges' 'pull ups' 'push ups' ...
        'rock climbing indoor' 'rope climbing' 'swing' 'tai chi' 'trampoline jumping' 'walking with a dog' 'wall pushups' ...
        'band marching' 'haircut' 'head massage' 'military parade' 'salsa spin' 'drumming' 'playing cello' 'playing daf' ...
        'playing dhol' 'playing flute' 'playing guitar' 'playing piano' 'playing sitar' 'playing tabla' 'playing violin' ...
        'archery' 'balance beam' 'baseball pitch' 'basketball' 'basketball dunk' 'bench press' 'biking' 'billiard' ...
        'bowling' 'boxing - punching bag' 'boxing - speed bag' 'breaststroke' 'clean and jerk' 'cliff diving' 'cricket bowling' ...
        'cricket shot' 'diving' 'fencing' 'field hockey penalty' 'floor gymnastics' 'frisbee catch' 'front crawl' ...
        'golf swing' 'hammer throw' 'high jump' 'horse race' 'horse riding' 'ice dancing' 'javeling throw' 'kayaking' ...
        'long jump' 'parallel bars' 'pole vault' 'pommel horse' 'punch' 'rafting' 'rowing' 'shotput' 'skiing' 'jetski' ...
        'sky diving' 'soccer penalty' 'stillrings' 'sumo wrestling' 'surfing' 'table tennis shot' 'tennis swing' ...
        'throw discuss' 'uneven bars' 'volleyball spiking'};
   
    tgt_labels_new = {'Apply Eye Makeup', 'Apply Lipstick', 'Archery', 'Baby Crawling', 'Balance Beam', 'Band Marching', ...
        'Baseball Pitch', 'Basketball', 'Basketball Dunk', 'Bench Press', 'Biking', 'Billiards Shot', ...
        'Blow Dry Hair', 'Blowing Candles', 'Body Weight Squats', 'Bowling', 'Boxing Punching Bag', 'Boxing Speed Bag', ...
        'Breaststroke', 'Brushing Teeth', 'Clean and Jerk', 'Cliff Diving', 'Cricket Bowling', 'Cricket Shot', ...
        'Cutting In Kitchen', 'Diving', 'Drumming', 'Fencing', 'Field Hockey Penalty', 'Floor Gymnastics', 'Frisbee Catch', ...
        'Front Crawl', 'Golf Swing', 'Haircut', 'Hammering', 'Hammer Throw', 'Handstand Pushups', 'Handstand Walking', ...
        'Head Massage', 'High Jump', 'Horse Race', 'Horse Riding', 'Hula Hoop', 'Ice Dancing', 'Javelin Throw', 'Juggling Balls', ...
        'Jumping Jack', 'Jump Rope', 'Kayaking', 'Knitting', 'Long Jump', 'Lunges', 'Military Parade', 'Mixing Batter', ...
        'Mopping Floor', 'Nun chucks', 'Parallel Bars', 'Pizza Tossing', 'Playing Cello', 'Playing Daf', 'Playing Dhol', ...
        'Playing Flute', 'Playing Guitar', 'Playing Piano', 'Playing Sitar', 'Playing Tabla', 'Playing Violin', 'Pole Vault', ...
        'Pommel Horse', 'Pull Ups', 'Punch', 'Push Ups', 'Rafting', 'Rock Climbing Indoor', 'Rope Climbing', 'Rowing', ...
        'Salsa Spins', 'Shaving Beard', 'Shotput', 'Skate Boarding', 'Skiing', 'Skijet', 'Sky Diving', 'Soccer Juggling', ...
        'Soccer Penalty', 'Still Rings', 'Sumo Wrestling', 'Surfing', 'Swing', 'Table Tennis Shot', 'Tai Chi', 'Tennis Swing', ...
        'Throw Discus', 'Trampoline Jumping', 'Typing', 'Uneven Bars', 'Volleyball Spiking', 'Walking with a dog', ...
        'Wall Pushups', 'Writing On Board', 'Yo Yo'};
    tgt_labels_new = lower(tgt_labels_new);
    
    if(strcmpi(phase,'source')) % Kinetics
        copy_labels = src_labels;
        % New config classes
        fprintf('[K=83] %s <- [U=25] %s \n', copy_labels{83}, tgt_labels_new{25});
        copy_labels{83} = tgt_labels_new{25};
        fprintf('[K=84] %s <- [U=25] %s \n', copy_labels{84}, tgt_labels_new{25});
        copy_labels{84} = tgt_labels_new{25};
        fprintf('[K=70] %s <- [U=46] %s \n', copy_labels{70}, tgt_labels_new{46});
        copy_labels{70} = tgt_labels_new{46};
        fprintf('[K=170] %s <- [U=46] %s \n', copy_labels{170}, tgt_labels_new{46});
        copy_labels{170} = tgt_labels_new{46};
        fprintf('[K=100] %s <- [U=8] %s \n', copy_labels{100}, tgt_labels_new{8});
        copy_labels{100} = tgt_labels_new{8};
        fprintf('[K=221] %s <- [U=8] %s \n', copy_labels{221}, tgt_labels_new{8});
        copy_labels{221} = tgt_labels_new{8};
        fprintf('[K=297] %s <- [U=8] %s \n', copy_labels{297}, tgt_labels_new{8});
        copy_labels{297} = tgt_labels_new{8};
        fprintf('[K=23] %s <- [U=11] %s \n', copy_labels{23}, tgt_labels_new{11});
        copy_labels{23} = tgt_labels_new{11};
        fprintf('[K=268] %s <- [U=11] %s \n', copy_labels{268}, tgt_labels_new{11});
        copy_labels{268} = tgt_labels_new{11};
        fprintf('[K=272] %s <- [U=11] %s \n', copy_labels{272}, tgt_labels_new{11});
        copy_labels{272} = tgt_labels_new{11};
        fprintf('[K=309] %s <- [U=81] %s \n', copy_labels{309}, tgt_labels_new{81});
        copy_labels{309} = tgt_labels_new{81};
        fprintf('[K=310] %s <- [U=81] %s \n', copy_labels{310}, tgt_labels_new{81});
        copy_labels{310} = tgt_labels_new{81};
        fprintf('[K=311] %s <- [U=81] %s \n', copy_labels{311}, tgt_labels_new{81});
        copy_labels{311} = tgt_labels_new{81};
        fprintf('[K=176] %s <- [U=85] %s \n', copy_labels{176}, tgt_labels_new{85});
        copy_labels{176} = tgt_labels_new{85};
        fprintf('[K=298] %s <- [U=85] %s \n', copy_labels{298}, tgt_labels_new{85});
        copy_labels{298} = tgt_labels_new{85};
        
        port_ids = copy_labels(ids)';
        
    elseif(strcmpi(phase,'target')) % UCF
        copy_labels = tgt_labels_new;
        
        % New config UCF
        fprintf('[U=1] %s <- [K=127] %s \n', copy_labels{1}, src_labels{127});
        copy_labels{1} = src_labels{127};
        fprintf('[U=20] %s <- [K=38] %s \n', copy_labels{20}, src_labels{38});
        copy_labels{20} = src_labels{38};
        fprintf('[U=43] %s <- [K=160] %s \n', copy_labels{43}, src_labels{160});
        copy_labels{43} = src_labels{160};
        fprintf('[U=48] %s <- [K=312] %s \n', copy_labels{48}, src_labels{312});
        copy_labels{48} = src_labels{312};
        fprintf('[U=50] %s <- [K=179] %s \n', copy_labels{50}, src_labels{179});
        copy_labels{50} = src_labels{179};
        fprintf('[U=55] %s <- [K=199] %s \n', copy_labels{55}, src_labels{199});
        copy_labels{55} = src_labels{199};
        fprintf('[U=58] %s <- [K=189] %s \n', copy_labels{58}, src_labels{189});
        copy_labels{58} = src_labels{189};
        fprintf('[U=78] %s <- [K=366] %s \n', copy_labels{78}, src_labels{366});
        copy_labels{78} = src_labels{366};
        fprintf('[U=80] %s <- [K=307] %s \n', copy_labels{80}, src_labels{307});
        copy_labels{80} = src_labels{307};
        fprintf('[U=84] %s <- [K=172] %s \n', copy_labels{84}, src_labels{172});
        copy_labels{84} = src_labels{172};
        fprintf('[U=4] %s <- [K=78] %s \n', copy_labels{4}, src_labels{78});
        copy_labels{4} = src_labels{78};
        fprintf('[U=14] %s <- [K=28] %s \n', copy_labels{14}, src_labels{28});
        copy_labels{14} = src_labels{28};
        fprintf('[U=15] %s <- [K=331] %s \n', copy_labels{15}, src_labels{331});
        copy_labels{15} = src_labels{331};
        fprintf('[U=52] %s <- [K=184] %s \n', copy_labels{52}, src_labels{184});
        copy_labels{52} = src_labels{184};
        fprintf('[U=70] %s <- [K=256] %s \n', copy_labels{70}, src_labels{256});
        copy_labels{70} = src_labels{256};
        fprintf('[U=72] %s <- [K=261] %s \n', copy_labels{72}, src_labels{261});
        copy_labels{72} = src_labels{261};
        fprintf('[U=74] %s <- [K=279] %s \n', copy_labels{74}, src_labels{279});
        copy_labels{74} = src_labels{279};
        fprintf('[U=75] %s <- [K=67] %s \n', copy_labels{75}, src_labels{67});
        copy_labels{75} = src_labels{67};
        fprintf('[U=89] %s <- [K=343] %s \n', copy_labels{89}, src_labels{343});
        copy_labels{89} = src_labels{343};
        fprintf('[U=91] %s <- [K=347] %s \n', copy_labels{91}, src_labels{347});
        copy_labels{91} = src_labels{347};
        fprintf('[U=94] %s <- [K=173] %s \n', copy_labels{94}, src_labels{173});
        copy_labels{94} = src_labels{173};
        fprintf('[U=98] %s <- [K=379] %s \n', copy_labels{98}, src_labels{379});
        copy_labels{98} = src_labels{379};
        fprintf('[U=99] %s <- [K=261] %s \n', copy_labels{99}, src_labels{261});
        copy_labels{99} = src_labels{261}; % mix 29
        fprintf('[U=6] %s <- [K=193] %s \n', copy_labels{6}, src_labels{193});
        copy_labels{6} = src_labels{193};
        fprintf('[U=34] %s <- [K=139] %s \n', copy_labels{34}, src_labels{139});
        copy_labels{34} = src_labels{139};
        fprintf('[U=39] %s <- [K=197] %s \n', copy_labels{39}, src_labels{197});
        copy_labels{39} = src_labels{197};
        fprintf('[U=77] %s <- [K=284] %s \n', copy_labels{77}, src_labels{284});
        copy_labels{77} = src_labels{284};
        fprintf('[U=27] %s <- [K=231] %s \n', copy_labels{27}, src_labels{231});
        copy_labels{27} = src_labels{231};
        fprintf('[U=59] %s <- [K=224] %s \n', copy_labels{59}, src_labels{224});
        copy_labels{59} = src_labels{224};
        fprintf('[U=62] %s <- [K=232] %s \n', copy_labels{62}, src_labels{232});
        copy_labels{62} = src_labels{232};
        fprintf('[U=63] %s <- [K=233] %s \n', copy_labels{63}, src_labels{233});
        copy_labels{63} = src_labels{233};
        fprintf('[U=64] %s <- [K=242] %s \n', copy_labels{64}, src_labels{242});
        copy_labels{64} = src_labels{242};
        fprintf('[U=67] %s <- [K=251] %s \n', copy_labels{67}, src_labels{251});
        copy_labels{67} = src_labels{251};
        fprintf('[U=3] %s <- [K=6] %s \n', copy_labels{3}, src_labels{6});
        copy_labels{3} = src_labels{6};
        fprintf('[U=7] %s <- [K=49] %s \n', copy_labels{7}, src_labels{49});
        copy_labels{7} = src_labels{49};
        fprintf('[U=9] %s <- [K=108] %s \n', copy_labels{9}, src_labels{108});
        copy_labels{9} = src_labels{108};
        fprintf('[U=10] %s <- [K=20] %s \n', copy_labels{10}, src_labels{20});
        copy_labels{10} = src_labels{20};
        fprintf('[U=16] %s <- [K=32] %s \n', copy_labels{16}, src_labels{32});
        copy_labels{16} = src_labels{32};
        fprintf('[U=17] %s <- [K=259] %s \n', copy_labels{17}, src_labels{259});
        copy_labels{17} = src_labels{259};
        fprintf('[U=18] %s <- [K=259] %s \n', copy_labels{18}, src_labels{259});
        copy_labels{18} = src_labels{259}; % mix 61
        fprintf('[U=19] %s <- [K=341] %s \n', copy_labels{19}, src_labels{341});
        copy_labels{19} = src_labels{341};
        fprintf('[U=21] %s <- [K=60] %s \n', copy_labels{21}, src_labels{60});
        copy_labels{21} = src_labels{60};
        fprintf('[U=22] %s <- [K=94] %s \n', copy_labels{22}, src_labels{94});
        copy_labels{22} = src_labels{94};
        fprintf('[U=23] %s <- [K=228] %s \n', copy_labels{23}, src_labels{228});
        copy_labels{23} = src_labels{228};
        fprintf('[U=24] %s <- [K=228] %s \n', copy_labels{24}, src_labels{228});
        copy_labels{24} = src_labels{228}; % mix 66
        fprintf('[U=26] %s <- [K=287] %s \n', copy_labels{26}, src_labels{287});
        copy_labels{26} = src_labels{287};
        fprintf('[U=31] %s <- [K=50] %s \n', copy_labels{31}, src_labels{50});
        copy_labels{31} = src_labels{50};
        fprintf('[U=33] %s <- [K=143] %s \n', copy_labels{33}, src_labels{143});
        copy_labels{33} = src_labels{143};
        fprintf('[U=36] %s <- [K=149] %s \n', copy_labels{36}, src_labels{149});
        copy_labels{36} = src_labels{149};
        fprintf('[U=40] %s <- [K=152] %s \n', copy_labels{40}, src_labels{152});
        copy_labels{40} = src_labels{152};
        fprintf('[U=42] %s <- [K=274] %s \n', copy_labels{42}, src_labels{274});
        copy_labels{42} = src_labels{274};
        fprintf('[U=45] %s <- [K=167] %s \n', copy_labels{45}, src_labels{167});
        copy_labels{45} = src_labels{167};
        fprintf('[U=49] %s <- [K=43] %s \n', copy_labels{49}, src_labels{43});
        copy_labels{49} = src_labels{43};
        fprintf('[U=51] %s <- [K=183] %s \n', copy_labels{51}, src_labels{183});
        copy_labels{51} = src_labels{183};
        fprintf('[U=68] %s <- [K=254] %s \n', copy_labels{68}, src_labels{254});
        copy_labels{68} = src_labels{254};
        fprintf('[U=71] %s <- [K=260] %s \n', copy_labels{71}, src_labels{260});
        copy_labels{71} = src_labels{260};
        fprintf('[U=79] %s <- [K=299] %s \n', copy_labels{79}, src_labels{299});
        copy_labels{79} = src_labels{299};
        fprintf('[U=82] %s <- [K=168] %s \n', copy_labels{82}, src_labels{168});
        copy_labels{82} = src_labels{168};
        fprintf('[U=83] %s <- [K=313] %s \n', copy_labels{83}, src_labels{313});
        copy_labels{83} = src_labels{313};
        fprintf('[U=88] %s <- [K=338] %s \n', copy_labels{88}, src_labels{338});
        copy_labels{88} = src_labels{338};
        fprintf('[U=92] %s <- [K=247] %s \n', copy_labels{92}, src_labels{247});
        copy_labels{92} = src_labels{247};
        fprintf('[U=93] %s <- [K=359] %s \n', copy_labels{93}, src_labels{359});
        copy_labels{99} = src_labels{359};
        fprintf('[U=97] %s <- [K=252] %s \n', copy_labels{97}, src_labels{252});
        copy_labels{97} = src_labels{252};

        port_ids = copy_labels(ids)';
    end
	new_labels = unique(copy_labels);
    
end
