# --------------------------------------------------
# AE functions for Camel dataset
# - data loading and pre-processing
# - data formatting
#
# alberto.bassi@eawag.ch
# --------------------------------------------------
includet("HydroNODE_data.jl")



function prepare_multiple_basins(data_path, source_data_set::String = "daymet")
    # function to load and preprocess camels data

    # ==========================================================================
    # check if basin data is incomplete, flawed, otherwise problematic...

    # flagged ids from CAMELS due to missing hydrologic years
    # (see "dataset_summary.txt" in "CAMELS time series meteorology, observed flow, meta data (.zip)"
    #  at https://ral.ucar.edu/solutions/products/camels  )

    basins_with_missing_data = lpad.([01208990, 01613050, 02051000, 02235200, 02310947,
        02408540,02464146,03066000,03159540,03161000,03187500,03281100,03300400,03450000,
        05062500,05087500,06037500,06043500,06188000,06441500,07290650,07295000,07373000,
        07376000,07377000,08025500,08155200,09423350,09484000,09497800,09505200,10173450,
        10258000,12025000,12043000,12095000,12141300,12374250,13310700],8,"0")

    # ==========================================================================
    # gauge_information has to be read first to obtain correct HUC (hydrologic unit code)
    path_gauge_meta = joinpath(data_path, "basin_metadata","gauge_information.txt")
    gauge_meta = CSV.File(path_gauge_meta, delim='\t', skipto=2, header = false)|> DataFrame

    all_basin_ids = lpad.(gauge_meta.Column2,8,"0")
    all_basin_huc = lpad.(gauge_meta.Column1,2,"0")
    
    # ==========================================================================
    # filter basins from the ones that miss data
    num_all_basin_ids = size(all_basin_ids)[1]
    num_basins_with_missing_data  = size(basins_with_missing_data)[1]
    
    no_missing_data_indexes = []
    for i in 1:num_all_basin_ids
        no_missing_data = true
        for j in 1:num_basins_with_missing_data
            if all_basin_ids[i]==basins_with_missing_data[j]
                no_missing_data = false
            end
        end
        if no_missing_data
            push!(no_missing_data_indexes, i)
        end
    end
    
    
    filtered_all_basin_ids = all_basin_ids[no_missing_data_indexes]
    num_basins = size(filtered_all_basin_ids)[1]
    effecive_num_basins = 598
        
    #print(filtered_all_basin_ids[27])
    #print("\n")
    # ==========================================================================
    # build dataset by running over all basins
    input_var_names = ["Daylight(h)", "Prec(mm/day)", "Tmean(C)"]
    output_var_name = "Flow(mm/s)"
    # select start and stop dates
    start_date = Date(1980,01,01)
    stop_date = Date(1985,09,30)
    # compute timepoints
    all_times = collect(start_date:Day(1):stop_date)
    timepoints = findall(x->x==start_date, all_times)[1]:findall(x->x==stop_date, all_times)[1]
    timepoints = collect((timepoints.-1.0).*1.0)
    seq_len = size(timepoints)[1]
    # initialize data containers
    data_x = Array{Float32}(undef, (seq_len, 3, 1, effecive_num_basins))
    data_y = Array{Float32}(undef, (seq_len, 1, effecive_num_basins))
    
    
    i = 1
    for j in 1:num_basins
        # retrieve basins data
        basin_id = filtered_all_basin_ids[j]
        df = load_data(basin_id, data_path)
        
        # drop unused cols
        select!(df, Not(Symbol("SWE(mm)")));
        select!(df, Not(Symbol("Tmax(C)")));
        select!(df, Not(Symbol("Tmin(C)")));

        # put only basins that starts on 01.01.1980
        if df[1, "Date"] == start_date 
            # format data
            basin_x, basin_y = prepare_data(df, (start_date, stop_date),input_var_names,output_var_name)
        
            ################################
            # insert some normalization here
            ################################
            
            # update data containers
            @assert(size(basin_x)==(seq_len, 3))
            @assert(size(basin_y)==(seq_len,))
            data_x[:,:,:, i] = basin_x
            data_y[:,:,i] = basin_y
            # update counter
            i += 1
        end
        
    end
    
    return data_x, data_y, timepoints
end




function Encoder(enc_space_dim)
    # Define the encoder and decoder networks
    encoder_features = Chain(
        Conv((4, 4), 1 => 32, leakyrelu; stride = 2, pad = 1),
        Conv((4, 4), 32 => 32, relu; stride = 2, pad = 1),
        Conv((4, 4), 32 => 32, relu; stride = 2, pad = 1),
        Flux.flatten,
        Dense(32 * 4 * 4, 256, relu),
        Dense(256, enc_space_dim, relu)
    )
    
    return encoder_μ, encoder_logvar, decoder
end




