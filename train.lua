require 'dataset-mnist'
require 'torch'
require 'optim'
require 'nn'
require 'image'

------------------------------------------
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

local classes = {'1','2','3','4','5','6','7','8','9','0'}

local geometry = {32,32}
------------------------------------------



--[[
--  YOUR CODE GOES HERE
--]]



------------------------------------------
local trainingPatches = 2000
local testingPatches = 1000

local trainData = mnist.loadTrainSet(trainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

testData = mnist.loadTestSet(testingPatches, geometry)
testData:normalizeGlobal(mean, std)

confusion = optim.ConfusionMatrix(classes)

function train(dataset)

    epoch = epoch or 1
    for t = 1,dataset:size(),10 do
    
        local inputs = torch.Tensor(10,1,geometry[1],geometry[2])
        local targets = torch.Tensor(10)
        local k = 1

        for i = t,math.min(t+9,dataset:size()) do
            
            local sample = dataset[i]
            local input = sample[1]:clone()
            local _,target = sample[2]:clone():max(1)
            target = target:squeeze()
            inputs[k] = input
            targets[k] = target
            k = k + 1
        end

------------------------------------------



--[[
--  YOUR CODE GOES HERE
--]]



------------------------------------------
    end

    print(confusion)
    confusion:zero()

    epoch = epoch + 1
end


function test(dataset)

    for t = 1,dataset:size(), 10 do

        xlua.progress(t, dataset:size())

        local inputs = torch.Tensor(10,1,geometry[1],geometry[2])
        local targets = torch.Tensor(10)
        local  k = 1

        for i = t, math.min(t+9, dataset:size()) do
            local sample = dataset[i]
            local input = sample[1]:clone()
            local _,target = sample[2]:clone():max(1)
            target = target:squeeze()
            inputs[k] = input
            targets[k] = target
            k = k + 1
        end
------------------------------------------



        --[[
        --  YOUR CODE GOES HERE
        --]]



------------------------------------------
end
------------------------------------------



--[[
--  YOUR CODE GOES HERE
--]]



