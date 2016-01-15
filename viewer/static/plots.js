function unique(arr){
    var o = {}, i, l = arr.length, r = [];
    for(i=0; i<l;i+=1) o[arr[i]] = arr[i];
    for(i in o) r.push(o[i]);
    return r;
};

function populate(data){
    for(var filter in data){
        $('#' + filter).empty();
        var qfilter = data[filter];
        if(filter == 'channel'){
            qfilter.sort(function(a, b){
                var ar = a.split("").reverse().join("");
                var br = b.split("").reverse().join("");
                return ar.localeCompare(br);
            });
        }
        if(filter == 'cv'){
            qfilter.sort(function(a, b){
                return parseInt(a) > parseInt(b);
            });
        }
        for(var i in qfilter){
            $('#' + filter).append(
                $('<option value="'+qfilter[i]+'">'+qfilter[i]+'</option>')
            );
        }
    }
}

function query(){
    var q = {};
    $('#filter select').each(function(){
        if((v = $(this).val()) != null)
            q[$(this).attr('id')] = v;
    });
    return q;
}

function namespace(){
    return $('#namespace').val();
}

function update(query){
    $('#filter select').prop('disabled', true);
    if(window.experiment_req != undefined){
        window.experiment_req.abort();
    }
    window.experiment_req = $.getJSON('experiments/' + namespace() + '/summary/' + JSON.stringify(query), function(data){
        populate(data);
        $('#filter select').prop('disabled', false);
        window.experiment_req = undefined;
    });
}

$(document).ready(function(){
    update(query());
    $('#namespace').change(function(){
        update({});
    });
    $('#filter select').change(function(){
        update(query());
    });
    $('#filter button').click(function(){
        var average = $('#mean').prop("checked");
        d3.json('experiments/' + namespace() + '/' + JSON.stringify(query()) + (average ? '?average' : ''), function(error, json) {
          if (error) throw error;
          var data = json['data'];
          render(data, average);
        });
    });
    $('#draw').highcharts({
        chart: { zoomType: 'x' },
        xAxis: { type: 'category' },
        title: false,
        tooltip: { crosshairs: true, shared: true, },
        legend: { enabled: true },
        plotOptions: { series: { marker: { enabled: false } } }
    });
});

function name(pipeline, config, experiment, model, fold, channel, i){
    var name = '';
    if(unique(pipeline).length > 1)
        name += 'Pipe: ' + pipeline[i];
    if(unique(config).length > 1)
        name += 'C: ' + config[i];
    if(unique(experiment).length > 1)
        name += 'Exp: ' + experiment[i];
    if(unique(model).length > 1)
        name += 'Model: ' + model[i];
    if(unique(fold).length > 1)
        name += 'Fold: ' + fold[i];
    if(unique(channel).length > 1)
        name += 'C: ' + channel[i];
    return name;
}

function prepareSeries(data, average){
    var series = [];

    var pipeline = [];
    var config = [];
    var experiment = [];
    var model = [];
    var fold = [];
    var channel = [];

    for(var d in data){
        var s = data[d].name.split('.');
        pipeline.push(s[0]);
        config.push(s[1]);
        experiment.push(s[2]);
        model.push(s[3]);
        fold.push(s[4]);
        channel.push(s[5]);
    }

    if(average){
        for(var d in data){
            var s = data[d];
            console.log(s)
            var fmean = {
                zIndex: 1,
                type: 'line',
                name: name(pipeline, config, experiment, model, fold, channel, parseInt(d)),
                data: [],
                marker: {
                    lineWidth: 2,
                    lineColor: Highcharts.getOptions().colors[parseInt(d)]
                }
            };
            var frange = {
                zIndex: 0,
                type: 'arearange',
                lineWidth: 0,
                linkedTo: ':previous',
                name: name(pipeline, config, experiment, model, fold, channel, parseInt(d)),
                data: [],
                color: Highcharts.getOptions().colors[parseInt(d)],
                fillOpacity: 0.3,
            };
            for(var v in s.range){
                var point = s.mean[v];
                fmean['data'].push([
                    point.epoch,
                    point.mean,
                ]);
                var point = s.range[v];
                frange['data'].push([
                    point.epoch,
                    point.min,
                    point.max,
                ]);
            }
            series.push(fmean);
            series.push(frange);
        }
    } else {
        for(var d in data){
            var s = data[d];
            var f = {
                type: 'line',
                name: name(pipeline, config, experiment, model, fold, channel, parseInt(d)),
                data: [],
            };
            for(var v in s.values){
                var point = s.values[v];
                f['data'].push([
                    point.epoch,
                    point.value,
                ]);
            }
            series.push(f);
        }
    }
    return series;
}

function render(data, average){

    var series = prepareSeries(data, average);

    function remove(arr, obj){
        var found = arr.indexOf(obj);
        if(found !== -1) arr.splice(found, 1);
    }

    var h = $('#draw').highcharts();
    var to_add = [];
    var to_remove = [];
    for(var i in series){
        var ns = series[i];
        to_add.push(ns.name);
    }
    for(var i in h.series){
        var s = h.series[i];
        to_remove.push(s.name);
    }
    for(var i in h.series){
        var s = h.series[i];
        for(var j in series){
            var ns = series[j];
            if(s.name == ns.name){
                remove(to_remove, s.name)
                remove(to_add, ns.name)
                s.setData(ns.data);
            }
        }
    }
    for(var i in to_remove){
        for(var j in h.series){
            var s = h.series[j];
            if(s.name == to_remove[i]){
                h.series[j].remove();
            }
        }
    }
    for(var j in series){
        var ns = series[j];
        if($.inArray(ns.name, to_add) > -1){
            h.addSeries(ns);
        }
    }
}