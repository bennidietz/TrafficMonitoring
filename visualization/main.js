"use strict";

  if (document.getElementById("visualization") !== null) {
    const visualization = new Vue({
      el: "#visualization",
      data: {
        csvData: null,
        isLoading: false,
      },
      methods: {
        readData: function(e) {
          if (!this.isLoading) {
            this.isLoading = true;

            if (window.FileReader && e.target.files[0]) {
              const file = e.target.files[0];
              const reader = new FileReader();

              reader.onload = function(e) {
                visualization.csvData = e.target.result;
              }

              reader.readAsText(file);
            }

            this.isLoading = false;
          }
        },
        drawCanvas: function() {
          var ctx = document.getElementById("canvas").getContext("2d");
          var chart = new Chart(ctx, {
            type: "bar",
            data: {
              labels: ["2020-12-07", "2020-12-08", "2020-12-09", "2020-12-10"],
              datasets: [
                {
                  label: "Cars",
                  backgroundColor: "rgb(255, 99, 132)",
                  borderColor: "rgb(255, 99, 132)",
                  data: [2, 1, 0, 2]
                },
                {
                  label: "Trucks",
                  backgroundColor: "rgb(255, 99, 0)",
                  borderColor: "rgb(255, 99, 0)",
                  data: [0, 0, 1, 1]
                },
              ]
            },
            options: {
              scales: {
                xAxes: [{
                  stacked: true
                }],
                yAxes: [{
                  stacked: true
                }]
              }
            }
          });
        },
      },
      computed: {
        csvValues: function() {
          var data = [];

          if (this.csvData) {
            const lines = this.csvData.split("\n");
            const headers = lines[0].split(",");
            var result = [];

            for (var i = 1; i < (lines.length - 1); i++) {
              var obj = {};
              var currentLine = lines[i].split(",");

              for (var j=0; j < headers.length; j++) {
                obj[headers[j]] = currentLine[j].replace("\r", "");
              }

              result.push(obj);
            }

            data = result;
          }

          return data;
        },
        csvTable: function() {
          var dataTable = "<table cellpadding='0' cellspacing='0'>";

          if (this.csvData) {
            dataTable = dataTable + "<tr>";

            const lines = this.csvData.split("\n");
            const headers = lines[0].split(",");

            for (var i = 0; i < headers.length; i++) {
              dataTable = dataTable + "<th>" + headers[i] + "</th>";
            }

            dataTable = dataTable + "</tr>";

            for (var i = 0; i < this.csvValues.length; i++) {
              dataTable = dataTable + "<tr>";

              for (var j = 0; j < headers.length; j++) {
                dataTable = dataTable + "<td>" + this.csvValues[i][headers[j]] + "</td>";
              }

              dataTable = dataTable + "</tr>";
            }
          }

          dataTable = dataTable + "</table>";

          this.drawCanvas();

          return dataTable;
        }
      }
    });
  }
