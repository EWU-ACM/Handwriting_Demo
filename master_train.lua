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

local model = nn.Sequential()

model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(3,3,3,3))

model:add(nn.SpatialConvolutionMM(32,64,5,5))
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(2,2,2,2))

model:add(nn.Reshape(64*2*2))
model:add(nn.Linear(64*2*2, 200))
model:add(nn.Tanh())
model:add(nn.Linear(200, 10))

local params, gradParams = model:getParameters()

model:add(nn.LogSoftMax())
local criterion = nn.ClassNLLCriterion()


------------------------------------------
local trainingPatches = 2000
local testingPatches = 1000

local trainData = mnist.loadTrainSet(trainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

testData = mnist.loadTestSet(testingPatches, geometry)
testData:normalizeGlobal(mean, std)

confusion = optim.ConfusionMatrix(classes)
------------------------------------------

--[[
--  YOUR CODE GOES HERE
--]]

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

        local feval = function(x)

            collectgarbage()

            if x ~= params then
                params:copy(x)
            end

            gradParams:zero()

            local outputs = model:forward(inputs)
            local f = criterion:forward(outputs, targets)
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do)

            for i = 1, 10 do
                confusion:add(outputs[i], targets[i])
            end

            return f, gradParams
        end

        state = state or {
                            learninRate = 0.05,
                            momentum = 0,
                            learningRateDecay = 5e-7
                        }

        optim.sgd(feval, params, state)

        xlua.progress(t, dataset:size())
    end

    print(confusion)
    confusion:zero()

    epoch = epoch + 1
end

------------------------------------------
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
        
        local preds = model:forward(inputs)

        for i = 1, 10 do
            confusion:add(preds[i], targets[i])
        end
    end
    
    print(confusion)
    confusion:zero()

------------------------------------------
end
------------------------------------------

--[[
--  YOUR CODE GOES HERE
--]]

for i = 1, 25 do

    print('Training\n\n')
    train(trainData)
    print('Testing\n\n')
    test(testData)

end
