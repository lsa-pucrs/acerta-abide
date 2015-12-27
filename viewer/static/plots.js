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
    $.getJSON('experiments/' + namespace() + '/summary/' + JSON.stringify(query), function(data){
        populate(data);
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
        d3.json('experiments/' + namespace() + '/' + JSON.stringify(query()), function(error, json) {
          if (error) throw error;
          var data = json['data'];
          render(data);
        });
    });
    $('#draw').highcharts({
        chart: { zoomType: 'x' },
        xAxis: { type: 'category' },
        title: false,
        legend: {
            enabled: true
        },
        plotOptions: {
            series: {
                marker: {
                    enabled: false
                }
            }
        }
    });
});

function render(data){

    series = [];
    for(var d in data){
        var s = data[d];

        var f = {
            type: 'line',
            name: s.name,
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
    console.log(to_add)
    console.log(to_remove);
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