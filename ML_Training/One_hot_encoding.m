

One_hot_matrix = [];
for row_num = 1:177

    This_row = strtrim(split(T.SkillsRequired{row_num},',')');
    row_hot_index = [];
    row_hot_vector = zeros(1,162);
    for i = 1:length(This_row)
        index = find(strcmp(unique_dict,This_row{i}));
        row_hot_index = [row_hot_index index];
    end
    
    for i = 1:length(row_hot_index)
        row_hot_vector(row_hot_index(i)) = 1;
    end

One_hot_matrix = [One_hot_matrix; row_hot_vector];

end
